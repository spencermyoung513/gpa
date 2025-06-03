import torch
from gpa.common.helpers import edge_index_union
from gpa.datasets.attribution import DetectionGraph
from scipy import stats
from torch_geometric.transforms import BaseTransform


class SampleRandomSubgraph(BaseTransform):
    """A transform that randomly forms a partial subgraph of a `DetectionGraph` (to be completed by a model)."""

    def __init__(self, a: float, b: float):
        """Create a `SampleRandomSubgraph` transform.

        Args:
            a (float): The alpha parameter of the beta distribution we sample p from (the probability of a UPC cluster being dropped).
            b (float): The beta parameter of the beta distribution we sample p from (the probability of a UPC cluster being dropped).
        """
        if a <= 0 or b <= 0:
            raise ValueError("Both a and b must be positive.")
        self.a = a
        self.b = b

    def forward(self, graph: DetectionGraph) -> DetectionGraph:
        """Apply the `SampleRandomSubgraph` transform to the given graph.

        This transformation will apply random Bernoulli dropout to a DetectionGraph's
        ground-truth prod-price edges (all price edges are dropped for a UPC group with
        probability `p`), where the dropout probability `p` is sampled from a beta
        distribution with parameters `a` and `b`.

        This subsample of the ground truth will then be unioned with the graph's
        existing edge index (which connects product nodes of the same UPC) to form a new
        subgraph.

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
        cluster_indices = graph.upc_clusters.unique()
        keep_cluster = torch.rand_like(cluster_indices, dtype=torch.float) > p
        keep_indices = torch.where(
            torch.isin(graph.upc_clusters, cluster_indices[keep_cluster])
            | torch.isin(torch.arange(graph.x.shape[0]), graph.price_indices)
        )[0]
        src_mask = torch.isin(graph.gt_prod_price_edge_index[0], keep_indices)
        dst_mask = torch.isin(graph.gt_prod_price_edge_index[1], keep_indices)
        sub_index = graph.gt_prod_price_edge_index[:, src_mask & dst_mask]

        new_graph = graph.clone()
        new_graph.edge_index = edge_index_union(graph.edge_index, sub_index)
        return new_graph
