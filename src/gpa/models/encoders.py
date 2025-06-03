import torch
from gpa.common.enums import EncoderType
from torch import nn
from torch.nn.functional import relu
from torch_geometric.nn import GATv2Conv
from torch_geometric.nn import TransformerConv
from torch_geometric.utils import scatter


class Encoder(nn.Module):
    """A neural network that encodes the nodes of a graph."""

    def __init__(self, node_hidden_dim: int, num_layers: int):
        """Initialize an encoder.

        Args:
            node_hidden_dim (int): The dimension of the hidden representation for each node.
            num_layers (int): The number of layers in the encoder.
        """
        super().__init__()
        self.node_hidden_dim = node_hidden_dim
        self.num_layers = num_layers

    @property
    def layers(self) -> torch.nn.ModuleList:
        raise NotImplementedError("Should be implemented by subclass.")

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.LongTensor,
        cluster_assignment: torch.LongTensor | None = None,
    ) -> torch.Tensor:
        """Encode the specified graph nodes.

        Args:
            x (torch.Tensor): Graph node embeddings, with shape (n, node_dim).
            edge_index (torch.LongTensor): Edge indices specifying the adjacency matrix for the graph being predicted on, with shape (2, num_edges).
            cluster_assignment (torch.LongTensor | None, optional): A vector assigning each node to a cluster, with shape (n,). If None, no clustering is used. Otherwise, after the encoding layers have been applied, we average the node embeddings within each cluster.

        Returns:
            torch.Tensor: The encoded node embeddings, with shape (n, `self.node_hidden_dim`).
        """
        for layer in self.layers[:-1]:
            x = relu(layer(x, edge_index=edge_index))
        x = self.layers[-1](x, edge_index=edge_index)
        if cluster_assignment is not None:
            cluster_mean = scatter(x, cluster_assignment, dim=0, reduce="mean")
            x = cluster_mean[cluster_assignment]
        return x


class TransformerEncoder(Encoder):
    """An encoder that uses a series of `TransformerConv` layers to encode the nodes of a graph."""

    def __init__(self, node_hidden_dim: int, num_layers: int):
        super().__init__(
            node_hidden_dim=node_hidden_dim,
            num_layers=num_layers,
        )
        self.convs = torch.nn.ModuleList(
            [TransformerConv(-1, node_hidden_dim) for _ in range(self.num_layers)]
        )

    @property
    def layers(self) -> torch.nn.ModuleList:
        return self.convs


class GATEncoder(Encoder):
    """An encoder that uses a series of `GATv2Conv` layers to encode the nodes of a graph."""

    def __init__(self, node_hidden_dim: int, num_layers: int):
        super().__init__(
            node_hidden_dim=node_hidden_dim,
            num_layers=num_layers,
        )
        self.convs = torch.nn.ModuleList(
            [GATv2Conv(-1, node_hidden_dim) for _ in range(self.num_layers)]
        )

    @property
    def layers(self) -> torch.nn.ModuleList:
        return self.convs


ENCODER_REGISTRY: dict[EncoderType, type[Encoder]] = {
    EncoderType.TRANSFORMER: TransformerEncoder,
    EncoderType.GAT: GATEncoder,
}
