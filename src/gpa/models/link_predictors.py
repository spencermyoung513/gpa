from typing import Literal

import torch
from gpa.common.enums import LinkPredictorType
from torch import nn


class LinkPredictor(nn.Module):
    """A neural network that predicts the likelihood of a link between a pair of nodes in a graph."""

    def forward(
        self,
        x: torch.Tensor,
        src: torch.Tensor,
        dst: torch.Tensor,
    ) -> torch.Tensor:
        """Predict the likelihood of a link between nodes for each pair indexed by src/dst.

        Args:
            x (torch.Tensor): Node embeddings, with shape (n, `self.node_hidden_dim`).
            src (torch.Tensor): Indices of source nodes in the pairs we are predicting links between, with shape (num_links_to_predict,).
            dst (torch.Tensor): Indices of destination nodes in the pairs we are predicting links between, with shape (num_links_to_predict,).

        Returns:
            torch.Tensor: The link logits, with shape (num_links_to_predict,).
        """
        raise NotImplementedError("Should be implemented by subclass.")


class MLPLinkPredictor(LinkPredictor):
    """A link predictor that uses a series of linear layers to predict link probability."""

    def __init__(
        self,
        layer_widths: list[int],
        use_batch_norm: bool = False,
        dropout: float | None = None,
        strategy: Literal["concat", "hadamard", "add"] = "hadamard",
    ):
        """Initialize an `MLPLinkPredictor`.

        Args:
            layer_widths (list[int]): The desired widths of the linear layers in the MLP.
            use_batch_norm (bool): Whether to use batch normalization between linear layers. Defaults to False.
            dropout (float): The dropout rate to use after each linear layer. Defaults to 0.0.
            strategy (Literal["concat", "hadamard", "add"]): The strategy used to combine the node embeddings.
        """
        super().__init__()

        self.strategy = strategy
        self.layers = nn.ModuleList()
        for i, width in enumerate(layer_widths):
            self.layers.append(nn.LazyLinear(out_features=width))
            if i < len(layer_widths) - 1:
                if use_batch_norm:
                    self.layers.append(nn.BatchNorm1d(width))
                self.layers.append(nn.ReLU())
                if dropout is not None:
                    self.layers.append(nn.Dropout(dropout))
        self.link_predictor = nn.LazyLinear(out_features=1)

    def forward(
        self,
        x: torch.Tensor,
        src: torch.Tensor,
        dst: torch.Tensor,
    ) -> torch.Tensor:
        if self.strategy == "concat":
            h = torch.cat([x[src], x[dst]], dim=1)
        elif self.strategy == "hadamard":
            h = x[src] * x[dst]
        elif self.strategy == "add":
            h = x[src] + x[dst]
        for layer in self.layers:
            h = layer(h)
        logits: torch.Tensor = self.link_predictor(h)
        return logits.squeeze(1)


class InnerProductLinkPredictor(LinkPredictor):
    """A link predictor that uses the inner product of normalized node embeddings to predict link probability."""

    def forward(
        self,
        x: torch.Tensor,
        src: torch.Tensor,
        dst: torch.Tensor,
    ) -> torch.Tensor:
        x_src = x[src] / x[src].norm(dim=1, keepdim=True)
        x_dst = x[dst] / x[dst].norm(dim=1, keepdim=True)
        return torch.sum(x_src * x_dst, dim=1)


LINK_PREDICTOR_REGISTRY: dict[LinkPredictorType, type[LinkPredictor]] = {
    LinkPredictorType.MLP: MLPLinkPredictor,
    LinkPredictorType.INNER_PRODUCT: InnerProductLinkPredictor,
}
