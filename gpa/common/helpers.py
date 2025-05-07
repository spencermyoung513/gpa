import matplotlib.patches as patches
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch_geometric.data import Batch

from gpa.common.constants import IS_PRICE
from gpa.common.constants import IS_PRODUCT
from gpa.common.objects import UndirectedGraph


def min_max_normalize(x: torch.Tensor) -> torch.Tensor:
    """Min-max normalize the given tensor."""
    return (x - x.min()) / (x.max() - x.min() + 1e-6)


def edge_index_diff(edge_index_a: torch.Tensor, edge_index_b: torch.Tensor) -> torch.Tensor:
    """Return the edge index containing only edges in `edge_index_a` that are not in `edge_index_b`."""
    set_a = set(map(tuple, edge_index_a.T.tolist()))
    set_b = set(map(tuple, edge_index_b.T.tolist()))
    diff_pairs = set_a - set_b

    if diff_pairs:
        diff_tensor = torch.tensor(list(diff_pairs)).T
    else:
        diff_tensor = torch.empty((2, 0), dtype=edge_index_a.dtype)

    return diff_tensor


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


def plot_bboxes(
    bboxes: torch.Tensor,
    ax: plt.Axes,
    linestyle: str = "solid",
    color: str | tuple[float, float, float] | None = None,
    width: float = 1.0,
    height: float = 1.0,
):
    """Plot the given array of bounding boxes on the provided axes.

    Args:
        bboxes (torch.Tensor): A (n, 4) tensor of bounding boxes in xywhn format.
        ax (plt.Axes): The axes to plot the boxes on.
        linestyle (str, optional): Linestyle for the bounding box (passed to matplotlib). Defaults to "solid".
        color (str | tuple[float, float, float] | None, optional): Edge color for the bounding box. Defaults to None.
        width (float): The desired width of the plot (bboxes will be rescaled from relative to abs. coordinates).
        height (float): The desired height of the plot (bboxes will be rescaled from relative to abs. coordinates).
    """
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.set_aspect("equal")

    for bbox in bboxes:
        x, y, w, h = bbox.flatten().tolist()
        x = x * width
        y = y * height
        w = w * width
        h = h * height
        rect = patches.Rectangle(
            (x - w / 2, y - h / 2),
            w,
            h,
            linewidth=1.5,
            edgecolor=color or "k",
            facecolor="none",
            linestyle=linestyle,
        )
        ax.add_patch(rect)


def parse_into_subgraphs(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """Parse the graph specified by `edge_index` into its disjoint subgraphs.

    Args:
        edge_index: The edge indices of the graph.
        num_nodes: The number of nodes in the graph.

    Returns:
        A tensor of node indices, where each index is the root of the subgraph it belongs to.
    """
    parent = torch.arange(num_nodes, device=edge_index.device)

    def find(u):
        while parent[u] != u:
            parent[u] = parent[parent[u]]
            u = parent[u]
        return u

    def union(u, v):
        u_root = find(u)
        v_root = find(v)
        if u_root != v_root:
            parent[v_root] = u_root

    src, dst = edge_index
    for u, v in zip(src.tolist(), dst.tolist()):
        union(u, v)

    for i in range(num_nodes):
        parent[i] = find(i)

    return parent


def connect_products_with_nearest_price_tag_below(
    centroids: torch.Tensor,
    product_indices: torch.LongTensor,
    price_indices: torch.LongTensor,
) -> torch.LongTensor:
    """From the provided centroids, return an edge index connecting each product with the nearest price tag below it (if one exists).

    Args:
        centroids (torch.Tensor): Bbox centroids of all nodes, with shape (n, d).
        product_indices (torch.LongTensor): Indices of product nodes.
        price_indices (torch.LongTensor): Indices of price tag nodes.

    Returns:
        torch.LongTensor: A (2, E) edge index connecting each product centroid with the nearest price tag below it (if one exists).
    """
    product_centroids = centroids[product_indices]
    price_centroids = centroids[price_indices]

    distances = torch.cdist(product_centroids, price_centroids, p=2)
    product_y = product_centroids[:, 1].unsqueeze(1)
    price_y = price_centroids[:, 1].unsqueeze(0)
    # Recall: with bbox coordinates, the top of an image is y=0.
    under_mask = price_y > product_y
    distances[~under_mask] = float("inf")

    idx_of_nearest = torch.argmin(distances, dim=1)
    any_valid = torch.isfinite(distances).any(dim=1)

    nearest_price_tensor = torch.full(
        (len(product_indices),), -1, dtype=torch.long, device=product_centroids.device
    )
    nearest_price_tensor[any_valid] = price_indices[idx_of_nearest[any_valid]]

    return torch.stack([product_indices[any_valid], nearest_price_tensor[any_valid]], dim=0)


def get_candidate_edges(batch: Batch, balanced: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
    """Given a batch of graphs (represented as a mega-graph with a block diagonal adjacency matrix), return a set of candidate edges (of which some are fake and some are present in the ground truth labels).

    Args:
        batch (Batch): A batch of data from a PriceAttributionDataset.
        balanced (bool): Whether to balance the number of real and fake edges that are returned.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: Tensors specifying real/fake edges in the mega-graph.
    """
    real_edges = []
    fake_edges = []
    scene_indices = sorted(batch.batch.unique().tolist())

    for scene_idx in scene_indices:

        # Get the global indices that correspond to the current scene.
        current_scene_indices = torch.where(batch.batch == scene_idx)[0]

        # From the saved data, retrieve the ground-truth prod-price edges in the current scene (to teach the model what to look for).
        src_in_scene = torch.isin(batch.gt_prod_price_edge_index[0], current_scene_indices)
        dst_in_scene = torch.isin(batch.gt_prod_price_edge_index[1], current_scene_indices)
        real_edges_in_scene: torch.Tensor = batch.gt_prod_price_edge_index[
            :, src_in_scene & dst_in_scene
        ]

        # Generate fake prod-price edges for the current scene (to teach the model what *not* to look for.)
        centroids = batch.x[current_scene_indices, :2]
        candidate_edges = connect_products_with_nearest_price_tag_below(
            centroids=centroids,
            product_indices=torch.where(batch.x[current_scene_indices, -1] == 1)[0],
            price_indices=torch.where(batch.x[current_scene_indices, -1] == 0)[0],
        )
        candidate_edges = torch.stack(
            [current_scene_indices[candidate_edges[0]], current_scene_indices[candidate_edges[1]]],
            dim=0,
        )
        candidate_edges = torch.cat([candidate_edges, candidate_edges.flip(0)], dim=1)
        fake_edges_in_scene = edge_index_diff(candidate_edges, real_edges_in_scene)

        if balanced:
            n_positive = real_edges_in_scene.shape[1]
            n_negative = fake_edges_in_scene.shape[1]
            if n_positive == 0 or n_negative == 0:
                continue
            n_samples = min(n_positive, n_negative)
            real_edges_in_scene = real_edges_in_scene[
                :, torch.multinomial(torch.ones(n_positive), n_samples, replacement=False)
            ]
            fake_edges_in_scene = fake_edges_in_scene[
                :, torch.multinomial(torch.ones(n_negative), n_samples, replacement=False)
            ]

        real_edges.append(real_edges_in_scene)
        fake_edges.append(fake_edges_in_scene)

    # Form a combined edge index for positive and negative edges across all scenes in the batch.
    real_edges = (
        torch.cat(real_edges, dim=-1) if real_edges else torch.empty((2, 0), dtype=torch.long)
    )
    fake_edges = (
        torch.cat(fake_edges, dim=-1) if fake_edges else torch.empty((2, 0), dtype=torch.long)
    )

    return real_edges, fake_edges
