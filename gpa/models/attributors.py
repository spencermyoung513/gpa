from typing import Literal

import lightning as L
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch_geometric.data import Batch
from torchmetrics.classification import BinaryF1Score
from torchmetrics.classification import BinaryPrecision
from torchmetrics.classification import BinaryRecall

from gpa.common.enums import EncoderType
from gpa.common.enums import LinkPredictorType
from gpa.common.helpers import get_candidate_edges
from gpa.datasets.attribution import DetectionGraph
from gpa.models.encoders import ENCODER_REGISTRY
from gpa.models.link_predictors import LINK_PREDICTOR_REGISTRY


class PriceAttributor(nn.Module):
    def __init__(
        self,
        encoder_type: EncoderType,
        encoder_settings: dict,
        link_predictor_type: LinkPredictorType,
        link_predictor_settings: dict,
    ):
        super().__init__()
        self.encoder = ENCODER_REGISTRY[encoder_type](**encoder_settings)
        self.link_predictor = LINK_PREDICTOR_REGISTRY[link_predictor_type](
            **link_predictor_settings
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        src: torch.Tensor,
        dst: torch.Tensor,
    ) -> torch.Tensor:
        """Predict the probability that the node pairs indexed by src/dst are connected (conditioned on the input graph).

        For a product / price tag pair, this is the probability that the product is sold at that price.

        Args:
            x (torch.Tensor): Node embeddings, with shape (n, node_dim).
            edge_index (torch.Tensor): Edge indices specifying the adjacency matrix for the graph being predicted on, with shape (2, num_edges).
            edge_attr (torch.Tensor): Edge attributes, with shape (num_edges, edge_dim).
            src (torch.Tensor): Indices of source nodes in the pairs we are predicting links between, with shape (num_links_to_predict,).
            dst (torch.Tensor): Indices of destination nodes in the pairs we are predicting links between, with shape (num_links_to_predict,).

        Returns:
            torch.Tensor: Logits representing the link probability for each pair, with shape (num_links_to_predict,).
        """
        h = self.encoder(x, edge_index, edge_attr)
        return self.link_predictor(h, src, dst)


class LightningPriceAttributor(L.LightningModule):
    def __init__(
        self,
        encoder_type: EncoderType,
        encoder_settings: dict,
        link_predictor_type: LinkPredictorType,
        link_predictor_settings: dict,
        num_epochs: int = 1,
        lr: float = 3e-4,
        weight_decay: float = 1e-5,
        balanced_edge_sampling: bool = True,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.num_epochs = num_epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.balanced_edge_sampling = balanced_edge_sampling
        self.model = PriceAttributor(
            encoder_type=encoder_type,
            encoder_settings=encoder_settings,
            link_predictor_type=link_predictor_type,
            link_predictor_settings=link_predictor_settings,
        )

        # For lazy initialization.
        self.example_input_array = {
            "x": torch.randn(3, DetectionGraph.NODE_DIM),
            "edge_index": torch.tensor([[0, 1, 2], [0, 1, 2]]),
            "edge_attr": torch.randn(3, DetectionGraph.EDGE_DIM),
            "src": torch.tensor([0, 1]),
            "dst": torch.tensor([1, 0]),
        }

        self.trn_precision = BinaryPrecision()
        self.trn_recall = BinaryRecall()
        self.trn_f1 = BinaryF1Score()
        self.val_precision = BinaryPrecision()
        self.val_recall = BinaryRecall()
        self.val_f1 = BinaryF1Score()

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        src: torch.Tensor,
        dst: torch.Tensor,
    ) -> torch.Tensor:
        """Predict the probability that the node pairs indexed by src/dst are connected (conditioned on the input graph).

        For a product / price tag pair, this is the probability that the product is sold at that price.

        Args:
            x (torch.Tensor): Node embeddings, with shape (n, node_dim).
            edge_index (torch.Tensor): Edge indices specifying the adjacency matrix for the graph being predicted on, with shape (2, num_edges).
            edge_attr (torch.Tensor): Edge attributes, with shape (num_edges, edge_dim).
            src (torch.Tensor): Indices of source nodes in the pairs we are predicting links between, with shape (num_links_to_predict,).
            dst (torch.Tensor): Indices of destination nodes in the pairs we are predicting links between, with shape (num_links_to_predict,).

        Returns:
            torch.Tensor: Logits representing the link probability for each pair, with shape (num_links_to_predict,).
        """
        return self.model(x, edge_index, edge_attr, src, dst)

    def training_step(self, batch: Batch, batch_idx: int):
        return self._step(batch, step_type="train")

    def validation_step(self, batch: Batch, batch_idx: int):
        return self._step(batch, step_type="val")

    def _step(self, batch: Batch, step_type: Literal["train", "val"]) -> torch.Tensor:
        """Handle all the necessary logic for a single training / validation forward pass.

        Args:
            batch (Batch): A batch of data from a PriceAttributionDataset.
            step_type (Literal["train", "val"]: Specifies which type of step we are taking.

        Returns:
            torch.Tensor: The loss accumulated during the forward pass.
        """
        if step_type == "train":
            precision = self.trn_precision
            recall = self.trn_recall
            f1 = self.trn_f1
        elif step_type == "val":
            precision = self.val_precision
            recall = self.val_recall
            f1 = self.val_f1
        else:
            raise NotImplementedError("Unsupported step_type passed to PriceAttributor._step")

        real_edges, fake_edges = get_candidate_edges(batch, balanced=self.balanced_edge_sampling)
        loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        if real_edges.size(1) > 0:
            pos_preds = self(
                x=batch.x,
                edge_index=batch.edge_index,
                edge_attr=batch.edge_attr,
                src=real_edges[0],
                dst=real_edges[1],
            )
            pos_targets = torch.ones_like(pos_preds)
            pos_loss = F.binary_cross_entropy_with_logits(pos_preds, pos_targets)
            loss = loss + pos_loss

            pos_probs = torch.sigmoid(pos_preds)
            precision.update(pos_probs, pos_targets)
            recall.update(pos_probs, pos_targets)
            f1.update(pos_probs, pos_targets)

        if fake_edges.size(1) > 0:
            neg_preds = self(
                x=batch.x,
                edge_index=batch.edge_index,
                edge_attr=batch.edge_attr,
                src=fake_edges[0],
                dst=fake_edges[1],
            )
            neg_targets = torch.zeros_like(neg_preds)
            neg_loss = F.binary_cross_entropy_with_logits(neg_preds, neg_targets)
            loss = loss + neg_loss

            neg_probs = torch.sigmoid(neg_preds)
            precision.update(neg_probs, neg_targets)
            recall.update(neg_probs, neg_targets)
            f1.update(neg_probs, neg_targets)

        self.log(
            f"{step_type}_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch.num_graphs,
        )
        self.log(
            f"{step_type}_precision",
            precision,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch.num_graphs,
        )
        self.log(
            f"{step_type}_recall",
            recall,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch.num_graphs,
        )
        self.log(f"{step_type}_f1", f1, on_epoch=True, batch_size=batch.num_graphs)

        return loss

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = CosineAnnealingLR(optimizer, T_max=self.num_epochs, eta_min=self.lr / 10)
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
        }
