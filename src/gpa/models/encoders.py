import inspect
from abc import ABC
from abc import abstractmethod
from typing import Generic
from typing import TypeVar

import torch
from gpa.common.enums import EncoderType
from torch import nn
from torch.nn.functional import relu
from torch_geometric.nn import GATv2Conv
from torch_geometric.nn import TransformerConv


T = TypeVar("T", bound=nn.Module)


class Encoder(nn.Module, ABC, Generic[T]):
    """A neural network that encodes the nodes of a graph."""

    def __init__(
        self,
        node_hidden_dim: int,
        num_layers: int,
    ):
        """Initialize an encoder.

        Args:
            node_hidden_dim (int): The dimension of the hidden representation for each node.
            num_layers (int): The number of layers to use in the encoder.
        """
        super().__init__()
        self.node_hidden_dim = node_hidden_dim
        self.layers = nn.ModuleList(
            [self.layer_type(-1, node_hidden_dim) for _ in range(num_layers)]
        )
        self._validate_layers()

    @property
    @abstractmethod
    def layer_type(self) -> type[T]:
        pass

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.LongTensor,
        edge_attr: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode the specified graph nodes.

        Args:
            x (torch.Tensor): Graph node embeddings, with shape (n, node_dim).
            edge_index (torch.LongTensor): Edge indices specifying the adjacency matrix for the graph being predicted on, with shape (2, num_edges).
            edge_attr (torch.Tensor | None, optional): A (num_edges, edge_dim) tensor of edge features (in 1d, these are edge weights). If `None`, a (num_edges, 1) tensor of ones will be used.

        Returns:
            torch.Tensor: The encoded node embeddings, with shape (n, `self.node_hidden_dim`).
        """
        if edge_attr is not None and edge_attr.size(0) != edge_index.size(1):
            raise ValueError(
                "edge_attr and edge_index must have the same number of edges"
            )
        if edge_attr is None:
            edge_attr = torch.ones(
                edge_index.shape[1], 1, device=x.device, dtype=x.dtype
            )
        for idx, layer in enumerate(self.layers):
            x = layer(x, edge_index=edge_index, edge_attr=edge_attr)
            if idx < len(self.layers) - 1:
                x = relu(x)

        return x

    def _validate_layers(self):
        """Ensure that each layer is of the correct type and supports `edge_attr`."""
        for i, layer in enumerate(self.layers):
            assert isinstance(layer, self.layer_type), (
                f"Layer {i} is type{type(layer).__name__}, expected {self.layer_type.__name__}"
            )
            fwd_params = inspect.signature(layer.forward).parameters
            if "edge_attr" not in fwd_params:
                raise ValueError(
                    f"Layer {i} does not accept 'edge_attr' in its forward call. "
                    "Please use a layer that does, such as a `TransformerConv`."
                )


class TransformerEncoder(Encoder[TransformerConv]):
    @property
    def layer_type(self):
        return TransformerConv


class GATEncoder(Encoder[GATv2Conv]):
    @property
    def layer_type(self):
        return GATv2Conv


class IdentityEncoder(Encoder[nn.Identity]):
    @property
    def layer_type(self):
        return nn.Identity

    def __init__(self, *args, **kwargs):
        super().__init__(0, 0)


ENCODER_REGISTRY: dict[EncoderType, type[Encoder]] = {
    EncoderType.TRANSFORMER: TransformerEncoder,
    EncoderType.GAT: GATEncoder,
    EncoderType.IDENTITY: IdentityEncoder,
}
