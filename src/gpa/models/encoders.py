import torch
from gpa.common.enums import EncoderType
from torch import nn
from torch.nn.functional import relu
from torch_geometric.nn import GATv2Conv
from torch_geometric.nn import GCNConv
from torch_geometric.nn import TransformerConv


class Encoder(nn.Module):
    """A neural network that encodes the nodes of a graph."""

    def __init__(
        self,
        node_hidden_dim: int,
        layers: nn.ModuleList,
    ):
        """Initialize an encoder.

        Args:
            node_hidden_dim (int): The dimension of the hidden representation for each node.
            layers (nn.ModuleList): The layers of the encoder.
        """
        super().__init__()
        self.node_hidden_dim = node_hidden_dim
        self.layers = layers

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.LongTensor,
    ) -> torch.Tensor:
        """Encode the specified graph nodes.

        Args:
            x (torch.Tensor): Graph node embeddings, with shape (n, node_dim).
            edge_index (torch.LongTensor): Edge indices specifying the adjacency matrix for the graph being predicted on, with shape (2, num_edges).
            cluster_assignment (torch.LongTensor | None, optional): A vector assigning each node to a cluster, with shape (n,). If None, no clustering is used. Otherwise, after the encoding layers have been applied, we average the node embeddings within each cluster.

        Returns:
            torch.Tensor: The encoded node embeddings, with shape (n, `self.node_hidden_dim`).
        """
        for idx, layer in enumerate(self.layers):
            x = layer(x, edge_index=edge_index)
            if idx < len(self.layers) - 1:
                x = relu(x)

        return x


class TransformerEncoder(Encoder):
    """An encoder that uses a series of `TransformerConv` layers to encode the nodes of a graph."""

    def __init__(self, node_hidden_dim: int, num_layers: int):
        """Initialize a `TransformerEncoder`.

        Args:
            node_hidden_dim (int): The desired dimension of the hidden representation for each node.
            num_layers (int): The desired number of layers in the encoder.
        """
        layers = nn.ModuleList(
            [TransformerConv(-1, node_hidden_dim) for _ in range(num_layers)]
        )
        super().__init__(
            node_hidden_dim=node_hidden_dim,
            layers=layers,
        )


class GATEncoder(Encoder):
    """An encoder that uses a series of `GATv2Conv` layers to encode the nodes of a graph."""

    def __init__(self, node_hidden_dim: int, num_layers: int):
        """Initialize a `GATEncoder`.

        Args:
            node_hidden_dim (int): The desired dimension of the hidden representation for each node.
            num_layers (int): The desired number of layers in the encoder.
        """
        layers = nn.ModuleList(
            [GATv2Conv(-1, node_hidden_dim) for _ in range(num_layers)]
        )
        super().__init__(
            node_hidden_dim=node_hidden_dim,
            layers=layers,
        )


class GCNEncoder(Encoder):
    """An encoder that uses a series of `GCNConv` layers to encode the nodes of a graph."""

    def __init__(self, node_hidden_dim: int, num_layers: int):
        """Initialize a `GCNEncoder`.

        Args:
            node_hidden_dim (int): The desired dimension of the hidden representation for each node.
            num_layers (int): The desired number of layers in the encoder.
        """
        layers = nn.ModuleList(
            [GCNConv(-1, node_hidden_dim) for _ in range(num_layers)]
        )
        super().__init__(
            node_hidden_dim=node_hidden_dim,
            layers=layers,
        )


class IdentityEncoder(Encoder):
    """An encoder that simply returns the input node embeddings."""

    def __init__(self):
        """Initialize an `IdentityEncoder`."""
        super().__init__(
            node_hidden_dim=0,
            layers=nn.ModuleList(),
        )


ENCODER_REGISTRY: dict[EncoderType, type[Encoder]] = {
    EncoderType.TRANSFORMER: TransformerEncoder,
    EncoderType.GAT: GATEncoder,
    EncoderType.GCN: GCNEncoder,
    EncoderType.IDENTITY: IdentityEncoder,
}
