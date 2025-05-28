from gpa.common.helpers import edge_index_union
from gpa.datasets.attribution import DetectionGraph
from scipy import stats
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import dropout_edge


class SampleRandomSubgraph(BaseTransform):
    """A transform that randomly forms a partial subgraph of a `DetectionGraph` (to be completed by a model)."""

    def __init__(self, a: float, b: float):
        """Create a `SampleRandomSubgraph` transform.

        Args:
            a (float): The alpha parameter of the beta distribution we sample p from (1 - p is a sparsity factor).
            b (float): The beta parameter of the beta distribution we sample p from (1 - p is a sparsity factor).
        """
        if a <= 0 or b <= 0:
            raise ValueError("Both a and b must be positive.")
        self.a = a
        self.b = b

    def forward(self, graph: DetectionGraph) -> DetectionGraph:
        """Apply the `SampleRandomSubgraph` transform to the given graph.

        This transformation will apply random Bernoulli dropout to a DetectionGraph's
        ground-truth prod-price edges, where the dropout probability is sampled
        from a zero-inflated beta distribution with parameters `a` and `b`. This subsample of the ground
        truth will then be unioned with the graph's existing edge index (which connects
        product nodes of the same UPC) to form a new subgraph.

        Args:
            graph (DetectionGraph): The graph to apply the transform to.

        Returns:
            DetectionGraph: A transformed version of the graph.
        """
        if not isinstance(graph, DetectionGraph):
            raise ValueError(
                "`SampleRandomSubgraph` can only be applied to `DetectionGraph` objects."
            )
        p = stats.beta.rvs(self.a, self.b) if stats.uniform.rvs() < 0.5 else 0.0
        sub_index, _ = dropout_edge(
            edge_index=graph.gt_prod_price_edge_index,
            p=1 - p,
            force_undirected=True,
        )
        new_graph = graph.clone()
        new_graph.edge_index = edge_index_union(graph.edge_index, sub_index)
        return new_graph
