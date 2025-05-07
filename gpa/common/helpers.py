import torch
import torch.nn.functional as F

from gpa.common.constants import IS_PRICE
from gpa.common.constants import IS_PRODUCT
from gpa.common.objects import UndirectedGraph


def min_max_normalize(x: torch.Tensor) -> torch.Tensor:
    return (x - x.min()) / (x.max() - x.min() + 1e-6)


def build_graph_from_detections(
    product_bboxes: dict[str, torch.Tensor],
    product_embeddings: dict[str, torch.Tensor],
    price_bboxes: dict[str, torch.Tensor],
    price_embeddings: dict[str, torch.Tensor],
) -> UndirectedGraph:
    """Build an undirected graph from the provided detections.

    Args:
        product_bboxes (dict[str, torch.Tensor]): Mapping from product bbox IDs to their respective coordinates (xywhn format).
        product_embeddings (dict[str, torch.Tensor]): Mapping from product bbox IDs to their respective visual embeddings.
        price_bboxes (dict[str, torch.Tensor]): Mapping from price tag bbox IDs to their respective coordinates (xywhn format).
        price_embeddings (dict[str, torch.Tensor]): Mapping from price tag bbox IDs to their respective visual embeddings.

    Raises:
        ValueError: If any bbox coordinates are not normalized (values must be between 0 and 1).

    Returns:
        UndirectedGraph: A graph representation of the provided detections.
    """
    if torch.any(torch.stack(list(product_bboxes.values())) > 1):
        raise ValueError("Product bbox coords must be normalized.")
    if torch.any(torch.stack(list(price_bboxes.values())) > 1):
        raise ValueError("Price tag bbox coords must be normalized.")

    n_prod = len(product_bboxes)
    n_price = len(price_bboxes)

    # Construct node embeddings matrix (x).
    prod_bbox_ids, prod_bbox_tensors = tuple(map(list, zip(*product_bboxes.items())))
    price_bbox_ids, price_bbox_tensors = tuple(map(list, zip(*price_bboxes.items())))
    all_bboxes = torch.stack(prod_bbox_tensors + price_bbox_tensors, dim=0)
    indicator = torch.cat(
        [
            torch.full((n_prod,), IS_PRODUCT),
            torch.full((n_price,), IS_PRICE),
        ],
        dim=0,
    )
    x = torch.cat([all_bboxes, indicator.view(-1, 1)], dim=1)

    # Construct edge index connecting all nodes to all other nodes (undirected, no self loops).
    node_indices = torch.arange(n_prod + n_price)
    edge_index = torch.cartesian_prod(node_indices, node_indices).T
    edge_index = edge_index[:, edge_index[0] != edge_index[1]]  # Delete self loops.

    # Compute spatial proximity scores.
    centroids = all_bboxes[:, :2]
    centroid_dists = torch.cdist(centroids, centroids)
    edge_spatial_proximity = 1 / (centroid_dists[edge_index[0], edge_index[1]] + 1e-6)
    edge_spatial_proximity = min_max_normalize(edge_spatial_proximity)

    # Compute visual proximity scores.
    prod_embeddings = torch.stack([product_embeddings[id] for id in prod_bbox_ids], dim=0)
    normed_prod_embeddings = F.normalize(prod_embeddings, dim=1)
    prod_cosine_similarities = normed_prod_embeddings @ normed_prod_embeddings.T
    prod_visual_proximity = (1 + prod_cosine_similarities) / 2
    edge_visual_proximity = torch.zeros_like(edge_spatial_proximity)
    prod_prod_edges = (edge_index[0] < n_prod) & (edge_index[1] < n_prod)
    edge_visual_proximity[prod_prod_edges] = prod_visual_proximity[
        edge_index[0, prod_prod_edges], edge_index[1, prod_prod_edges]
    ]
    price_embeddings = torch.stack([price_embeddings[id] for id in price_bbox_ids], dim=0)
    normed_price_embeddings = F.normalize(price_embeddings, dim=1)
    price_cosine_similarities = normed_price_embeddings @ normed_price_embeddings.T
    price_visual_proximity = (1 + price_cosine_similarities) / 2
    price_price_edges = (edge_index[0] >= n_prod) & (edge_index[1] >= n_prod)
    edge_visual_proximity[price_price_edges] = price_visual_proximity[
        edge_index[0, price_price_edges] - n_prod,
        edge_index[1, price_price_edges] - n_prod,
    ]

    # Combine spatial / visual proximity scores into 2-d edge attributes.
    edge_attr = torch.stack([edge_spatial_proximity, edge_visual_proximity], dim=1)

    # Get a map that allows us to look up where a bounding box is in the node tensor.
    id_to_idx = {bbox_id: idx for idx, bbox_id in enumerate(prod_bbox_ids + price_bbox_ids)}

    return UndirectedGraph(id_to_idx=id_to_idx, x=x, edge_index=edge_index, edge_attr=edge_attr)
