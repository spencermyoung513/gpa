import torch
from torch import nn
from torch.nn.functional import relu
from torch_geometric.nn import GATv2Conv
from torch_geometric.nn import TransformerConv

from gpa.common.enums import EncoderType


class Encoder(nn.Module):
    def __init__(self, node_hidden_dim: int, edge_hidden_dim: int, num_layers: int):
        super().__init__()
        self.node_hidden_dim = node_hidden_dim
        self.edge_hidden_dim = edge_hidden_dim
        self.num_layers = num_layers

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        """Encode the specified graph nodes.

        Args:
            x (torch.Tensor): Graph node embeddings, with shape (n, node_dim).
            edge_index (torch.Tensor): Edge indices specifying the adjacency matrix for the graph being predicted on, with shape (2, num_edges).
            edge_attr (torch.Tensor): Edge attributes, with shape (num_edges, edge_dim).

        Returns:
            torch.Tensor: The encoded node embeddings, with shape (n, `self.node_hidden_dim`).
        """
        raise NotImplementedError("Should be implemented by subclass.")


class TransformerEncoder(Encoder):
    def __init__(self, node_hidden_dim: int, edge_hidden_dim: int, num_layers: int):
        super().__init__(
            node_hidden_dim=node_hidden_dim,
            edge_hidden_dim=edge_hidden_dim,
            num_layers=num_layers,
        )
        self.convs = torch.nn.ModuleList(
            [
                TransformerConv(-1, node_hidden_dim, edge_dim=edge_hidden_dim)
                for _ in range(self.num_layers)
            ]
        )
        self.edge_embedder = nn.LazyLinear(out_features=edge_hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        edge_h = self.edge_embedder(edge_attr)
        for conv in self.convs[:-1]:
            x = relu(conv(x, edge_index=edge_index, edge_attr=edge_h))
        x = self.convs[-1](x, edge_index=edge_index, edge_attr=edge_h)
        return x


class GATEncoder(Encoder):
    def __init__(self, node_hidden_dim: int, edge_hidden_dim: int, num_layers: int):
        super().__init__(
            node_hidden_dim=node_hidden_dim,
            edge_hidden_dim=edge_hidden_dim,
            num_layers=num_layers,
        )
        self.convs = torch.nn.ModuleList(
            [
                GATv2Conv(-1, node_hidden_dim, edge_dim=edge_hidden_dim)
                for _ in range(self.num_layers)
            ]
        )
        self.edge_embedder = nn.LazyLinear(out_features=edge_hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        edge_h = self.edge_embedder(edge_attr)
        for conv in self.convs[:-1]:
            x = relu(conv(x, edge_index=edge_index, edge_weight=edge_h))
        x = self.convs[-1](x, edge_index=edge_index, edge_weight=edge_h)
        return x


ENCODER_REGISTRY: dict[EncoderType, type[Encoder]] = {
    EncoderType.TRANSFORMER: TransformerEncoder,
    EncoderType.GAT: GATEncoder,
}
