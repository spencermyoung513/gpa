import torch
from gpa.common.enums import EncoderType
from torch import nn
from torch.nn.functional import relu
from torch_geometric.nn import GATv2Conv
from torch_geometric.nn import TransformerConv


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

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """Encode the specified graph nodes.

        Args:
            x (torch.Tensor): Graph node embeddings, with shape (n, node_dim).
            edge_index (torch.Tensor): Edge indices specifying the adjacency matrix for the graph being predicted on, with shape (2, num_edges).

        Returns:
            torch.Tensor: The encoded node embeddings, with shape (n, `self.node_hidden_dim`).
        """
        raise NotImplementedError("Should be implemented by subclass.")


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

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        for conv in self.convs[:-1]:
            x = relu(conv(x, edge_index=edge_index))
        x = self.convs[-1](x, edge_index=edge_index)
        return x


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

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        for conv in self.convs[:-1]:
            x = relu(conv(x, edge_index=edge_index))
        x = self.convs[-1](x, edge_index=edge_index)
        return x


ENCODER_REGISTRY: dict[EncoderType, type[Encoder]] = {
    EncoderType.TRANSFORMER: TransformerEncoder,
    EncoderType.GAT: GATEncoder,
}
