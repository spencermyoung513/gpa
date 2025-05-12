from __future__ import annotations

from typing import Callable

import matplotlib.pyplot as plt
import torch
from matplotlib import colormaps
from torch_geometric.data import Data
from torch_geometric.data import InMemoryDataset
from tqdm import tqdm

from src.common.constants import IS_PRICE
from src.common.constants import IS_PRODUCT
from src.common.helpers import build_graph_from_detections
from src.common.objects import GraphComponents
from src.common.objects import ProductPriceGroup
from src.common.objects import UPCGroup


class DetectionGraph(Data):
    """A graph of product and price tag detections."""

    NODE_DIM = 5  # (x, y, w, h, is_product)
    EDGE_DIM = 2  # (spatial_proximity, visual_proximity)
    INDICATOR_IDX = -1  # Index of the is_product indicator in the node embeddings.

    # Base attributes (part of the base `Data` class)
    x: torch.Tensor
    edge_index: torch.LongTensor
    edge_attr: torch.Tensor

    # Custom attributes
    graph_id: str
    bbox_ids: list[str]
    product_indices: torch.LongTensor
    price_indices: torch.LongTensor
    shared_upc_edge_index: torch.LongTensor
    gt_prod_price_edge_index: torch.LongTensor

    @classmethod
    def build(cls, graph_components: GraphComponents) -> DetectionGraph:
        """Assemble the provided components into a `DetectionGraph`.

        The node embeddings are made up of the (x, y, w, h) bbox coordinates and an indicator of if a node is a product or a price tag.

        The edge index fully connects the initial graph. The edge attributes contain a dimension for spatial proximity (normalized inverse euclidean distance)
        and a dimension for visual proximity (only relevant for prod-prod edges) obtained from cosine similarity.

        Args:
            graph_components (GraphComponents): The components of the graph to assemble.
        """
        base_graph = build_graph_from_detections(
            product_bboxes=graph_components.product_bboxes,
            product_embeddings=graph_components.product_embeddings,
            price_bboxes=graph_components.price_bboxes,
            price_embeddings=graph_components.price_embeddings,
        )
        shared_upc_edge_index = cls._get_shared_upc_edge_index(
            id_to_idx=base_graph.id_to_idx,
            upc_groups=graph_components.upc_groups,
        )
        gt_prod_price_edge_index = cls._get_gt_prod_price_edge_index(
            id_to_idx=base_graph.id_to_idx,
            prod_price_groups=graph_components.prod_price_groups,
        )
        product_mask = base_graph.x[:, cls.INDICATOR_IDX] == IS_PRODUCT
        price_mask = base_graph.x[:, cls.INDICATOR_IDX] == IS_PRICE
        product_indices = torch.argwhere(product_mask).flatten()
        price_indices = torch.argwhere(price_mask).flatten()

        bbox_ids = [None] * len(base_graph.id_to_idx)
        for bbox_id, idx in base_graph.id_to_idx.items():
            bbox_ids[idx] = bbox_id

        return cls(
            x=base_graph.x,
            edge_index=base_graph.edge_index,
            edge_attr=base_graph.edge_attr,
            graph_id=graph_components.graph_id,
            bbox_ids=bbox_ids,
            product_indices=product_indices,
            price_indices=price_indices,
            shared_upc_edge_index=shared_upc_edge_index,
            gt_prod_price_edge_index=gt_prod_price_edge_index,
        )

    def plot(
        self,
        ax: plt.Axes | None = None,
        prod_price_only: bool = False,
        mark_wrong_edges: bool = True,
    ) -> plt.Axes:
        """Return a visual representation of this graph.

        Product centroids will be plotted in blue, with price tag centroids in orange.

        Args:
            ax (plt.Axes | None, optional): The axes to plot on (if provided). Uses plt.gca() if None (default).
            prod_price_only (bool, optional): Whether to only plot product-price connections. Defaults to False (will also show product-product edges).
            mark_wrong_edges (bool, optional): Whether to mark incorrect (according to `self.gt_prod_price_edge_index`) product-price connections in red. Defaults to True.
        """
        good_cmap = colormaps["Blues"]
        bad_cmap = colormaps["Reds"]
        norm = plt.Normalize(0, 1)

        if ax is None:
            ax = plt.gca()

        product_centroids = self.x[self.product_indices, :2].view(-1, 2)
        price_centroids = self.x[self.price_indices, :2].view(-1, 2)
        gt_prod_price_edges = list(
            zip(
                self.gt_prod_price_edge_index[0].tolist(),
                self.gt_prod_price_edge_index[1].tolist(),
            )
        )

        ax.scatter(*product_centroids.T, zorder=10, c="tab:blue")
        ax.scatter(*price_centroids.T, zorder=10, c="tab:orange")

        src_is_product = torch.isin(self.edge_index[0], self.product_indices)
        src_is_price = torch.isin(self.edge_index[0], self.price_indices)
        dst_is_product = torch.isin(self.edge_index[1], self.product_indices)
        dst_is_price = torch.isin(self.edge_index[1], self.price_indices)
        is_prod_price_edge = (src_is_product & dst_is_price) | (src_is_price & dst_is_product)

        for j in range(self.edge_index.shape[1]):
            weight = self.edge_attr[j].norm(p=2)
            src = self.edge_index[0, j]
            dst = self.edge_index[1, j]
            if prod_price_only and not is_prod_price_edge[j]:
                continue
            if (
                mark_wrong_edges
                and is_prod_price_edge[j]
                and (src, dst) not in gt_prod_price_edges
            ):
                edge_color = bad_cmap(norm(weight))
            else:
                edge_color = good_cmap(norm(weight))
            ax.plot(
                [self.x[src, 0], self.x[dst, 0]],
                [self.x[src, 1], self.x[dst, 1]],
                color=edge_color,
                lw=0.5,
            )
        ax.set_ylim(0, 1)
        ax.set_xlim(0, 1)
        ax.invert_yaxis()
        ax.set_aspect("equal")
        ax.set_title(self.graph_id, fontsize=10)
        return ax

    @classmethod
    def _get_shared_upc_edge_index(
        cls,
        id_to_idx: dict[str, int],
        upc_groups: list[UPCGroup],
    ) -> torch.LongTensor:
        """Form an undirected edge index that indicates which nodes in the graph share the same UPC.

        Args:
            id_to_idx (dict[str, int]): A mapping from bbox IDs to their respective indices in the graph's node embeddings matrix.
            upc_groups (list[UPCGroup]): A list of UPC groups, where each group contains the bbox IDs of nodes that share the same UPC.

        Returns:
            torch.LongTensor: A (2, num_edges) tensor of edge indices.
        """
        edge_indices = []
        for group in upc_groups:
            group_indices = torch.tensor([id_to_idx[bbox_id] for bbox_id in group.bbox_ids])
            if group_indices.numel() == 0:
                continue
            edge_indices.append(torch.cartesian_prod(group_indices, group_indices).T)
        if not edge_indices:
            return torch.empty((2, 0), dtype=torch.long)
        return torch.cat(edge_indices, dim=1)

    @classmethod
    def _get_gt_prod_price_edge_index(
        cls,
        id_to_idx: dict[str, int],
        prod_price_groups: list[ProductPriceGroup],
    ) -> torch.LongTensor:
        """Form an undirected edge index that indicates which nodes in the graph are part of the same pricing group (according to ground truth labels).

        Args:
            id_to_idx (dict[str, int]): A mapping from bbox IDs to their respective indices in the graph's node embeddings matrix.
            prod_price_groups (list[ProductPriceGroup]): A list of product-price groups, where each group contains the product and price tag bbox IDs for a ground truth product-price group.

        Returns:
            torch.LongTensor: A (2, num_edges) tensor of edge indices.
        """
        edge_indices = []
        for group in prod_price_groups:
            product_indices = torch.tensor(
                [id_to_idx[bbox_id] for bbox_id in group.product_bbox_ids]
            )
            price_indices = torch.tensor([id_to_idx[bbox_id] for bbox_id in group.price_bbox_ids])
            if product_indices.numel() == 0 or price_indices.numel() == 0:
                continue
            one_way = torch.cartesian_prod(product_indices, price_indices).T
            edge_indices.append(torch.cat([one_way, one_way.flip(0)], dim=1))
        if not edge_indices:
            return torch.empty((2, 0), dtype=torch.long)
        return torch.cat(edge_indices, dim=1)

    def __cat_dim__(self, key, value, *args, **kwargs):
        if key in ("product_indices", "price_indices"):
            return 0
        return super().__cat_dim__(key, value, *args, **kwargs)

    def __inc__(self, key, value, *args, **kwargs):
        if key in ("product_indices", "price_indices"):
            return self.num_nodes
        return super().__inc__(key, value, *args, **kwargs)


torch.serialization.add_safe_globals([DetectionGraph])


class PriceAttributionDataset(InMemoryDataset):
    """A dataset of price attribution graphs, where each graph ."""

    def __init__(
        self,
        root: str,
        transform: Callable[[DetectionGraph], DetectionGraph] | None = None,
        **kwargs,
    ) -> None:
        """Initialize a `PriceAttributionDataset` dataset.

        Args:
            root (str): Root directory where the dataset is stored.
            transform (Callable | None, optional): Transform to apply to each graph in the dataset when `__getitem__` is called. Defaults to None.
        """
        super().__init__(root, transform=transform, **kwargs)
        self.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ["graph_components.pt", "product_boxes.csv", "price_boxes.csv"]

    @property
    def processed_file_names(self):
        return ["data.pt"]

    def process(self):
        raw_graph_components = torch.load(self.raw_paths[0], weights_only=True)
        data_list = [
            DetectionGraph.build(GraphComponents(**g)) for g in tqdm(raw_graph_components)
        ]
        self.save(data_list, self.processed_paths[0])
