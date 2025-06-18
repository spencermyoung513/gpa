import torch
from gpa.common.enums import ConnectionStrategy
from gpa.common.helpers import connect_products_with_nearest_price_tag
from gpa.common.helpers import connect_products_with_nearest_price_tag_below
from gpa.common.helpers import connect_products_with_nearest_price_tag_per_group
from gpa.common.helpers import edge_index_union
from gpa.datasets.attribution import DetectionGraph
from gpa.models.attributors import PriceAttributor
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
        on whether/not the graph is clustered by UPC (path 1) or not (path 2). In both cases,
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
        if graph.get("upc_clusters") is not None:
            sub_index = self._cluster_dropout(graph, p)
        else:
            sub_index = self._edge_dropout(graph, p)

        new_graph = graph.clone()
        new_graph.edge_index = edge_index_union(graph.edge_index, sub_index)
        return new_graph

    def _cluster_dropout(self, graph: DetectionGraph, p: float) -> torch.Tensor:
        assert graph.get("upc_clusters") is not None
        cluster_indices = graph.upc_clusters.unique()
        keep_cluster = torch.rand_like(cluster_indices, dtype=torch.float) > p
        keep_indices = torch.where(
            torch.isin(graph.upc_clusters, cluster_indices[keep_cluster])
            | torch.isin(torch.arange(graph.x.shape[0]), graph.price_indices)
        )[0]
        src_mask = torch.isin(graph.gt_prod_price_edge_index[0], keep_indices)
        dst_mask = torch.isin(graph.gt_prod_price_edge_index[1], keep_indices)
        sub_index = graph.gt_prod_price_edge_index[:, src_mask & dst_mask]
        return sub_index

    def _edge_dropout(self, graph: DetectionGraph, p: float) -> torch.Tensor:
        assert graph.get("upc_clusters") is None
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
            edge_index = connect_products_with_nearest_price_tag_per_group(
                centroids=graph.x[:, graph.BBOX_START_IDX : graph.BBOX_START_IDX + 2],
                product_indices=graph.product_indices,
                price_indices=graph.price_indices,
                cluster_assignment=graph.upc_clusters,
            )
        else:
            raise ValueError(f"Unknown connection strategy: {self.strategy}")
        new_graph = graph.clone()
        new_graph.edge_index = edge_index_union(graph.edge_index, edge_index)
        return new_graph


class ConnectGraphWithModel(BaseTransform):
    """A transform that connects the nodes of a `DetectionGraph` using a trained model."""

    def __init__(self, model: PriceAttributor, edge_prob_threshold: float = 0.5):
        """Initialize a `ConnectGraphWithModel` transform.

        Args:
            model (PriceAttributor): The model to connect detection graphs with.
            edge_prob_threshold (float, optional): Link probability threshold for connecting two nodes. Set to 0 to connect everything. Defaults to 0.5.
        """
        self.model = model
        self.edge_prob_threshold = edge_prob_threshold

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
        x = graph.x
        edge_index = graph.edge_index
        cluster_assignment = graph.get("upc_clusters")
        src, dst = torch.cartesian_prod(graph.product_indices, graph.price_indices).T
        probs = self.model(
            x=x,
            edge_index=edge_index,
            src=src,
            dst=dst,
            cluster_assignment=cluster_assignment,
        ).sigmoid()
        prod_price_edge_index = torch.stack([src, dst], dim=0)
        prod_prod_mask = torch.isin(
            graph.edge_index[0], graph.product_indices
        ) & torch.isin(graph.edge_index[1], graph.product_indices)
        prod_prod_edge_index = graph.edge_index[:, prod_prod_mask]

        keep_mask = probs > self.edge_prob_threshold

        prod_price_edge_attr = probs[keep_mask].view(-1, 1)
        prod_price_edge_index = prod_price_edge_index[:, keep_mask]
        prod_prod_edge_attr = torch.ones((prod_prod_edge_index.shape[1], 1))
        new_graph = graph.clone()
        new_graph.edge_index = torch.cat(
            [prod_price_edge_index, prod_prod_edge_index], dim=1
        )
        new_graph.edge_attr = torch.cat(
            [prod_price_edge_attr, prod_prod_edge_attr], dim=0
        )
        return new_graph
