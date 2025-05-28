import matplotlib.patches as patches
import matplotlib.pyplot as plt
import torch
from gpa.common.constants import IS_PRICE
from gpa.common.constants import IS_PRODUCT
from torch_geometric.data import Batch


def min_max_normalize(x: torch.Tensor) -> torch.Tensor:
    """Min-max normalize the given tensor."""
    return (x - x.min()) / (x.max() - x.min() + 1e-6)


def edge_index_union(
    edge_index_a: torch.Tensor, edge_index_b: torch.Tensor
) -> torch.Tensor:
    """Return an edge index containing all unique edges in `edge_index_a` and `edge_index_b`."""
    set_a = set(map(tuple, edge_index_a.T.tolist()))
    set_b = set(map(tuple, edge_index_b.T.tolist()))

    if unioned_pairs := set_a.union(set_b):
        union_tensor = torch.tensor(list(unioned_pairs)).T
    else:
        union_tensor = torch.empty((2, 0), dtype=edge_index_a.dtype)

    return union_tensor


def edge_index_diff(
    edge_index_a: torch.Tensor, edge_index_b: torch.Tensor
) -> torch.Tensor:
    """Return an edge index containing only edges in `edge_index_a` that are not in `edge_index_b`."""
    set_a = set(map(tuple, edge_index_a.T.tolist()))
    set_b = set(map(tuple, edge_index_b.T.tolist()))
    diff_pairs = set_a - set_b

    if diff_pairs:
        diff_tensor = torch.tensor(list(diff_pairs)).T
    else:
        diff_tensor = torch.empty((2, 0), dtype=edge_index_a.dtype)

    return diff_tensor


def get_node_embeddings_from_detections(
    product_bboxes: dict[str, torch.Tensor],
    product_embeddings: dict[str, torch.Tensor],
    price_bboxes: dict[str, torch.Tensor],
    price_embeddings: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, dict[str, int]]:
    """Build an undirected graph from the provided detections.

    Args:
        product_bboxes (dict[str, torch.Tensor]): Mapping from product bbox IDs to their respective coordinates (xywhn format).
        product_embeddings (dict[str, torch.Tensor]): Mapping from product bbox IDs to their respective embeddings.
        price_bboxes (dict[str, torch.Tensor]): Mapping from price tag bbox IDs to their respective coordinates (xywhn format).
        price_embeddings (dict[str, torch.Tensor]): Mapping from price tag bbox IDs to their respective embeddings.

    Raises:
        ValueError: If any bbox coordinates are not normalized (values must be between 0 and 1).

    Returns:
        tuple[torch.Tensor, dict[str, int]]: A tuple containing the node embeddings matrix and a map from each bbox ID to its corresponding row in the tensor.
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
    all_embeddings = torch.stack(
        list(product_embeddings.values()) + list(price_embeddings.values()), dim=0
    )
    indicator = torch.cat(
        [
            torch.full((n_prod,), IS_PRODUCT),
            torch.full((n_price,), IS_PRICE),
        ],
        dim=0,
    )
    x = torch.cat([all_bboxes, all_embeddings, indicator.view(-1, 1)], dim=1)

    # Get a map that allows us to look up where a bounding box is in the node tensor.
    id_to_idx = {
        bbox_id: idx for idx, bbox_id in enumerate(prod_bbox_ids + price_bbox_ids)
    }
    return x, id_to_idx


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


def connect_products_with_nearest_price_tag(
    centroids: torch.Tensor,
    product_indices: torch.LongTensor,
    price_indices: torch.LongTensor,
) -> torch.LongTensor:
    """From the provided centroids, return an edge index connecting each product with the nearest price tag.

    Args:
        centroids (torch.Tensor): Bbox centroids of all nodes, with shape (n, d).
        product_indices (torch.LongTensor): Indices of product nodes.
        price_indices (torch.LongTensor): Indices of price tag nodes.

    Returns:
        torch.LongTensor: A (2, E) edge index connecting each product centroid with the nearest price tag.
    """
    product_centroids = centroids[product_indices]
    price_centroids = centroids[price_indices]
    distances = torch.cdist(product_centroids, price_centroids, p=2)
    idx_of_nearest = torch.argmin(distances, dim=1)
    nearest_price_tensor = price_indices[idx_of_nearest]
    return torch.stack([product_indices, nearest_price_tensor], dim=0)


def get_candidate_edges(
    batch: Batch, balanced: bool = True
) -> tuple[torch.Tensor, torch.Tensor]:
    """Given a batch of graphs (represented as a mega-graph with a block diagonal adjacency matrix),
    return a set of candidate edges, of which some are fake and some are present in the ground truth.

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
        gt_src_in_scene = torch.isin(
            batch.gt_prod_price_edge_index[0], current_scene_indices
        )
        gt_dst_in_scene = torch.isin(
            batch.gt_prod_price_edge_index[1], current_scene_indices
        )
        real_edges_in_scene: torch.Tensor = batch.gt_prod_price_edge_index[
            :, gt_src_in_scene & gt_dst_in_scene
        ]
        # Remove edges that are already present (trivial positives).
        current_src_in_scene = torch.isin(batch.edge_index[0], current_scene_indices)
        current_dst_in_scene = torch.isin(batch.edge_index[1], current_scene_indices)
        real_edges_in_scene_not_already_present = edge_index_diff(
            real_edges_in_scene,
            batch.edge_index[:, current_src_in_scene & current_dst_in_scene],
        )

        # Generate fake prod-price edges for the current scene (to teach the model what *not* to look for.)
        product_indices = torch.where(batch.x[current_scene_indices, -1] == 1)[0]
        price_indices = torch.where(batch.x[current_scene_indices, -1] == 0)[0]
        candidate_edges = torch.cartesian_prod(product_indices, price_indices).T
        candidate_edges = torch.stack(
            [
                # We have to shift the indices to the global (mega-graph) frame of reference.
                current_scene_indices[candidate_edges[0]],
                current_scene_indices[candidate_edges[1]],
            ],
            dim=0,
        )
        # Make undirected (just in case the model is not symmetric).
        candidate_edges = torch.cat([candidate_edges, candidate_edges.flip(0)], dim=1)
        fake_edges_in_scene = edge_index_diff(candidate_edges, real_edges_in_scene)

        if balanced:
            n_positive = real_edges_in_scene_not_already_present.shape[1]
            n_negative = fake_edges_in_scene.shape[1]
            if n_positive == 0 or n_negative == 0:
                continue
            n_samples = min(n_positive, n_negative)
            real_edges_in_scene_not_already_present = (
                real_edges_in_scene_not_already_present[
                    :,
                    torch.multinomial(
                        torch.ones(n_positive), n_samples, replacement=False
                    ),
                ]
            )
            fake_edges_in_scene = fake_edges_in_scene[
                :,
                torch.multinomial(torch.ones(n_negative), n_samples, replacement=False),
            ]

        real_edges.append(real_edges_in_scene_not_already_present)
        fake_edges.append(fake_edges_in_scene)

    # Form a combined edge index for positive and negative edges across all scenes in the batch.
    real_edges = (
        torch.cat(real_edges, dim=-1)
        if real_edges
        else torch.empty((2, 0), dtype=torch.long)
    )
    fake_edges = (
        torch.cat(fake_edges, dim=-1)
        if fake_edges
        else torch.empty((2, 0), dtype=torch.long)
    )

    return real_edges, fake_edges
