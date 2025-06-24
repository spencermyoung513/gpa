from collections import defaultdict
from pathlib import Path

import torch
from gpa.common.enums import ConnectionStrategy
from gpa.common.helpers import connect_products_with_nearest_price_tag
from gpa.common.helpers import connect_products_with_nearest_price_tag_below
from gpa.common.helpers import connect_products_with_nearest_price_tag_per_group
from gpa.common.helpers import edge_index_union
from gpa.common.helpers import parse_into_subgraphs
from gpa.datasets.attribution import DetectionGraph
from gpa.models.attributors import LightningPriceAttributor
from scipy import stats
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import dropout_edge


class MaskOutVisualInformation(BaseTransform):
    """A transform that masks out the visual information from a `DetectionGraph`."""

    def forward(self, graph: DetectionGraph) -> DetectionGraph:
        """Apply the `MaskOutVisualInformation` transform to the given graph.

        Args:
            graph (DetectionGraph): The graph to apply the transform to.

        Returns:
            DetectionGraph: A transformed version of the graph.
        """
        if not isinstance(graph, DetectionGraph):
            raise ValueError(
                "`MaskOutVisualInformation` can only be applied to `DetectionGraph` objects."
            )
        new_graph = graph.clone()
        bbox_coords = new_graph.x[
            :, new_graph.BBOX_START_IDX : new_graph.BBOX_END_IDX + 1
        ]
        indicator = new_graph.x[:, new_graph.INDICATOR_IDX]
        new_x = torch.cat([bbox_coords, indicator.view(-1, 1)], dim=1)
        new_graph.x = new_x
        return new_graph


class SampleRandomSubgraph(BaseTransform):
    """A transform that randomly forms a partial subgraph of a `DetectionGraph` (to be completed by a model)."""

    def __init__(self, a: float, b: float):
        """Create a `SampleRandomSubgraph` transform.

        Args:
            a (float): The alpha parameter of the beta distribution we sample the dropout probability from.
            b (float): The beta parameter of the beta distribution we sample the dropout probability from.
        """
        self.a = a
        self.b = b

    def forward(self, graph: DetectionGraph) -> DetectionGraph:
        """Apply the `SampleRandomSubgraph` transform to the given graph.

        This transformation will apply random Bernoulli dropout to a `DetectionGraph`'s
        ground-truth prod-price edges. The exact behavior of this dropout is conditional
        on whether/not the graph is grouped by UPC (path 1) or not (path 2). In both cases,
        we sample `p` from Beta(`a`, `b`).

        Path 1: For a given UPC group, with probability `p`, all corresponding ground-truth
        product-price edges from the group are dropped.

        Path 2: For each ground-truth product-price edge, with probability `p`, the edge
        is dropped.

        The remaining edges are then unioned with the graph's existing edge index (which
        simply connects product nodes of the same UPC) to form a new subgraph of the ground
        truth.

        Args:
            graph (DetectionGraph): The graph to apply the transform to.

        Returns:
            DetectionGraph: A transformed version of the graph.
        """
        if not isinstance(graph, DetectionGraph):
            raise ValueError(
                "`SampleRandomSubgraph` can only be applied to `DetectionGraph` objects."
            )
        p = stats.beta.rvs(self.a, self.b)
        if graph.get("shared_upc_edge_index") is not None:
            sub_index = self._group_dropout(graph, p)
        else:
            sub_index = self._edge_dropout(graph, p)

        new_graph = graph.clone()
        new_graph.edge_index = edge_index_union(graph.edge_index, sub_index)
        return new_graph

    def _group_dropout(self, graph: DetectionGraph, p: float) -> torch.Tensor:
        assert graph.get("shared_upc_edge_index") is not None
        upc_groups = parse_into_subgraphs(
            graph.shared_upc_edge_index, num_nodes=len(graph.x)
        )
        group_indices = upc_groups.unique()
        keep_group = torch.rand_like(group_indices, dtype=torch.float) > p
        keep_indices = torch.where(
            torch.isin(upc_groups, group_indices[keep_group])
            | torch.isin(torch.arange(graph.x.shape[0]), graph.price_indices)
        )[0]
        src_mask = torch.isin(graph.gt_prod_price_edge_index[0], keep_indices)
        dst_mask = torch.isin(graph.gt_prod_price_edge_index[1], keep_indices)
        sub_index = graph.gt_prod_price_edge_index[:, src_mask & dst_mask]
        return sub_index

    def _edge_dropout(self, graph: DetectionGraph, p: float) -> torch.Tensor:
        assert graph.get("shared_upc_edge_index") is None
        sub_index, _ = dropout_edge(
            edge_index=graph.gt_prod_price_edge_index,
            p=p,
            force_undirected=True,
        )
        return sub_index


class MakeBoundingBoxTranslationInvariant(BaseTransform):
    """A transform that converts the bounding box coordinates of a `DetectionGraph` into a translation-invariant spatial encoding."""

    def forward(self, graph: DetectionGraph) -> DetectionGraph:
        """Apply the `MakeBoundingBoxTranslationInvariant` transform to the given graph.

        Args:
            graph (DetectionGraph): The graph to apply the transform to.

        Returns:
            DetectionGraph: A transformed version of the graph.
        """
        if not isinstance(graph, DetectionGraph):
            raise ValueError(
                "`MakeBoundingBoxTranslationInvariant` can only be applied to `DetectionGraph` objects."
            )
        centroids = graph.x[:, graph.BBOX_START_IDX : graph.BBOX_START_IDX + 2]
        center = centroids.mean(dim=0)
        invariant_centroids = centroids - center
        new_graph = graph.clone()
        new_x = torch.cat(
            [invariant_centroids, graph.x[:, graph.BBOX_START_IDX + 2 :]], dim=1
        )
        new_graph.x = new_x
        return new_graph


class HeuristicallyConnectGraph(BaseTransform):
    """A transform that heuristically connects the nodes of a `DetectionGraph` (according to a preset scheme)."""

    def __init__(self, strategy: ConnectionStrategy):
        self.strategy = strategy

    def forward(self, graph: DetectionGraph) -> DetectionGraph:
        """Apply the `HeuristicallyConnectGraph` transform to the given graph.

        This transform connects product/price nodes within a graph according to a preset scheme, such as
        "connect each product to the nearest price tag" or "connect each product to the nearest price tag *below* it".
        It then forms a new edge index from these connections and merges it with the graph's existing edge
        index (which should specify which nodes are the same UPC).

        Args:
            graph (DetectionGraph): The graph to apply the transform to.

        Returns:
            DetectionGraph: A transformed version of the graph.
        """
        if not isinstance(graph, DetectionGraph):
            raise ValueError(
                "`HeuristicallyConnectGraph` can only be applied to `DetectionGraph` objects."
            )
        if self.strategy == ConnectionStrategy.NEAREST:
            edge_index = connect_products_with_nearest_price_tag(
                centroids=graph.x[:, graph.BBOX_START_IDX : graph.BBOX_START_IDX + 2],
                product_indices=graph.product_indices,
                price_indices=graph.price_indices,
            )
        elif self.strategy == ConnectionStrategy.NEAREST_BELOW:
            edge_index = connect_products_with_nearest_price_tag_below(
                centroids=graph.x[:, graph.BBOX_START_IDX : graph.BBOX_START_IDX + 2],
                product_indices=graph.product_indices,
                price_indices=graph.price_indices,
            )
        elif self.strategy == ConnectionStrategy.NEAREST_BELOW_PER_GROUP:
            upc_groups = parse_into_subgraphs(
                graph.shared_upc_edge_index, num_nodes=len(graph.x)
            )
            edge_index = connect_products_with_nearest_price_tag_per_group(
                centroids=graph.x[:, graph.BBOX_START_IDX : graph.BBOX_START_IDX + 2],
                product_indices=graph.product_indices,
                price_indices=graph.price_indices,
                cluster_assignment=upc_groups,
            )
        else:
            raise ValueError(f"Unknown connection strategy: {self.strategy}")
        new_graph = graph.clone()
        new_graph.edge_index = edge_index_union(graph.edge_index, edge_index)
        return new_graph


class ConnectGraphWithSeedModel(BaseTransform):
    """A transform that connects the nodes of a `DetectionGraph` using a pre-trained "seed" model."""

    def __init__(
        self,
        seed_model_chkp_path: Path,
        seed_model_transform: BaseTransform,
        edge_prob_threshold: float = 0.5,
    ):
        """Initialize a `ConnectGraphWithModel` transform.

        Args:
            seed_model_chkp_path (Path): Path to the weights of the seed model that will connect the graph.
            seed_model_transform (BaseTransform): Transform to call on graphs before passing through the seed model.
            edge_prob_threshold (float, optional): Link probability threshold for connecting two nodes according to model beliefs. Set to 0 to connect everything. Defaults to 0.5.
        """
        self.transform = seed_model_transform
        self.edge_prob_threshold = edge_prob_threshold
        self.model = LightningPriceAttributor.load_from_checkpoint(
            checkpoint_path=seed_model_chkp_path, map_location="cpu"
        ).model
        self._cache = defaultdict(dict)

    @torch.inference_mode()
    def forward(self, graph: DetectionGraph) -> DetectionGraph:
        """Apply the `ConnectGraphWithModel` transform to the given graph.

        This transform uses a trained model to connect the nodes of a `DetectionGraph`.

        Args:
            graph (DetectionGraph): The graph to apply the transform to.

        Returns:
            DetectionGraph: The graph, with an edge index connecting all products and price tags above the threshold (with edge attributes indicating the predicted edge probabilities).
        """
        if not isinstance(graph, DetectionGraph):
            raise ValueError(
                "`ConnectGraphWithModel` can only be applied to `DetectionGraph` objects."
            )

        if graph.graph_id in self._cache:
            new_edge_index = self._cache[graph.graph_id]["edge_index"]
            new_edge_attr = self._cache[graph.graph_id]["edge_attr"]
        else:
            inference_graph: DetectionGraph = self.transform(graph.clone())
            x = inference_graph.x
            edge_index = inference_graph.edge_index
            src, dst = (
                x.flatten()
                for x in torch.meshgrid(
                    inference_graph.product_indices,
                    inference_graph.price_indices,
                    indexing="ij",
                )
            )
            probs = self.model(
                x=x,
                edge_index=edge_index,
                edge_attr=inference_graph.get("edge_attr"),
                src=src,
                dst=dst,
            ).sigmoid()

            keep_mask = probs > self.edge_prob_threshold
            prod_price_edge_attr = probs[keep_mask].view(-1, 1)
            prod_price_edge_index = torch.stack([src, dst], dim=0)[:, keep_mask]

            src_is_product = torch.isin(
                inference_graph.edge_index[0],
                inference_graph.product_indices,
            )
            dst_is_product = torch.isin(
                inference_graph.edge_index[1],
                inference_graph.product_indices,
            )
            prod_prod_mask = src_is_product & dst_is_product
            prod_prod_edge_index = inference_graph.edge_index[:, prod_prod_mask]
            prod_prod_edge_attr = torch.ones((prod_prod_edge_index.shape[1], 1))

            new_edge_index = torch.cat(
                [prod_price_edge_index, prod_prod_edge_index], dim=1
            )
            new_edge_attr = torch.cat(
                [prod_price_edge_attr, prod_prod_edge_attr], dim=0
            )

            self._cache[graph.graph_id]["edge_index"] = new_edge_index
            self._cache[graph.graph_id]["edge_attr"] = new_edge_attr

        new_graph = graph.clone()
        new_graph.edge_index = new_edge_index
        new_graph.edge_attr = new_edge_attr
        return new_graph


class FilterExtraneousPriceTags(BaseTransform):
    def __init__(self):
        self._cache = {}

    def forward(self, graph: DetectionGraph) -> DetectionGraph:
        """Apply the `FilterExtraneousPriceTags` transform to the given graph.

        Args:
            graph (DetectionGraph): The graph to apply the transform to.

        Returns:
            DetectionGraph: A transformed version of the graph.
        """
        if not isinstance(graph, DetectionGraph):
            raise ValueError(
                "`FilterExtraneousPriceTags` can only be applied to `DetectionGraph` objects."
            )
        if graph.graph_id in self._cache:
            node_indices = self._cache[graph.graph_id]
        else:
            product_cx = graph.x[graph.product_indices, graph.BBOX_START_IDX]
            product_w = graph.x[graph.product_indices, graph.BBOX_START_IDX + 2]
            products_x_left = (product_cx - product_w / 2).min()
            products_x_right = (product_cx + product_w / 2).max()

            prices_cx = graph.x[graph.price_indices, graph.BBOX_START_IDX]
            prices_w = graph.x[graph.price_indices, graph.BBOX_START_IDX + 2]
            prices_x_min = prices_cx - prices_w / 2
            prices_x_max = prices_cx + prices_w / 2

            in_bounds = (prices_x_max >= products_x_left) & (
                prices_x_min <= products_x_right
            )
            filtered_price_indices = graph.price_indices[in_bounds]
            node_indices = torch.cat([graph.product_indices, filtered_price_indices])
            self._cache[graph.graph_id] = node_indices

        return DetectionGraph.subgraph(
            original_graph=graph, indices_to_keep=node_indices
        )
