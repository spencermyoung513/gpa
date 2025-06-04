from typing import Literal

import lightning as L
import torch
import torch.nn.functional as F
from gpa.common.enums import EncoderType
from gpa.common.enums import LinkPredictorType
from gpa.common.helpers import get_candidate_edges
from gpa.models.encoders import ENCODER_REGISTRY
from gpa.models.link_predictors import LINK_PREDICTOR_REGISTRY
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch_geometric.data import Batch
from torchmetrics.classification import BinaryF1Score
from torchmetrics.classification import BinaryPrecision
from torchmetrics.classification import BinaryRecall
from torchmetrics.wrappers import BootStrapper


class PriceAttributor(nn.Module):
    """A neural network that encodes a graph of object detections and predicts which products and prices share the is-priced-at relation."""

    def __init__(
        self,
        encoder_type: EncoderType,
        encoder_settings: dict,
        link_predictor_type: LinkPredictorType,
        link_predictor_settings: dict,
    ):
        """Initialize a `PriceAttributor`.

        Args:
            encoder_type (EncoderType): The type of encoder to use.
            encoder_settings (dict): The settings for the encoder.
            link_predictor_type (LinkPredictorType): The type of link predictor to use.
            link_predictor_settings (dict): The settings for the link predictor.
        """
        super().__init__()
        self.encoder = ENCODER_REGISTRY[encoder_type](**encoder_settings)
        self.link_predictor = LINK_PREDICTOR_REGISTRY[link_predictor_type](
            **link_predictor_settings
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        src: torch.Tensor,
        dst: torch.Tensor,
        cluster_assignment: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Predict the probability that the node pairs indexed by src/dst are connected (conditioned on the input graph).

        For a product / price tag pair, this is the probability that the product is sold at that price.

        Args:
            x (torch.Tensor): Node embeddings, with shape (n, node_dim).
            edge_index (torch.Tensor): Edge indices specifying the adjacency matrix for the graph being predicted on, with shape (2, num_edges).
            src (torch.Tensor): Indices of source nodes in the pairs we are predicting links between, with shape (num_links_to_predict,).
            dst (torch.Tensor): Indices of destination nodes in the pairs we are predicting links between, with shape (num_links_to_predict,).
            cluster_assignment (torch.Tensor | None, optional): A vector assigning each node to a cluster (e.g. a UPC group), with shape (n,). If None, no clustering is assumed.

        Returns:
            torch.Tensor: Logits representing the link probability for each pair, with shape (num_links_to_predict,).
        """
        h = self.encoder(
            x=x,
            edge_index=edge_index,
            cluster_assignment=cluster_assignment,
        )
        return self.link_predictor(x=h, src=src, dst=dst)


class LightningPriceAttributor(L.LightningModule):
    """A `LightningModule` wrapper for training / evaluating a `PriceAttributor`."""

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
        """Initialize a `LightningPriceAttributor`.

        Args:
            encoder_type (EncoderType): The type of encoder to use.
            encoder_settings (dict): The settings for the encoder.
            link_predictor_type (LinkPredictorType): The type of link predictor to use.
            link_predictor_settings (dict): The settings for the link predictor.
            num_epochs (int): The number of epochs to train for.
            lr (float): The learning rate.
            weight_decay (float): The weight decay.
            balanced_edge_sampling (bool): Whether to sample edges in a balanced manner (w.r.t. number of real/fake edges presented to the link predictor during training).
        """
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
        self.trn_precision = BootStrapper(BinaryPrecision())
        self.trn_recall = BootStrapper(BinaryRecall())
        self.trn_f1 = BootStrapper(BinaryF1Score())
        self.val_precision = BootStrapper(BinaryPrecision())
        self.val_recall = BootStrapper(BinaryRecall())
        self.val_f1 = BootStrapper(BinaryF1Score())
        self.test_precision = BootStrapper(BinaryPrecision())
        self.test_recall = BootStrapper(BinaryRecall())
        self.test_f1 = BootStrapper(BinaryF1Score())

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        src: torch.Tensor,
        dst: torch.Tensor,
        cluster_assignment: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Predict the probability that the node pairs indexed by src/dst are connected (conditioned on the input graph).

        For a product / price tag pair, this is the probability that the product is sold at that price.

        Args:
            x (torch.Tensor): Node embeddings, with shape (n, node_dim).
            edge_index (torch.Tensor): Edge indices specifying the adjacency matrix for the graph being predicted on, with shape (2, num_edges).
            src (torch.Tensor): Indices of source nodes in the pairs we are predicting links between, with shape (num_links_to_predict,).
            dst (torch.Tensor): Indices of destination nodes in the pairs we are predicting links between, with shape (num_links_to_predict,).
            cluster_assignment (torch.Tensor | None, optional): A (n,) tensor assigning each node to a cluster (e.g. a UPC group). Before link prediction, all node embeddings within a cluster will be average-pooled. If None, no clustering is assumed.

        Returns:
            torch.Tensor: Logits representing the link probability for each specified pair, with shape (num_links_to_predict,).
        """
        return self.model(
            x=x,
            edge_index=edge_index,
            src=src,
            dst=dst,
            cluster_assignment=cluster_assignment,
        )

    def training_step(self, batch: Batch, batch_idx: int):
        return self._step(batch, step_type="train")

    def validation_step(self, batch: Batch, batch_idx: int):
        return self._step(batch, step_type="val")

    def test_step(self, batch: Batch, batch_idx: int):
        return self._step(batch, step_type="test")

    def _step(
        self, batch: Batch, step_type: Literal["train", "val", "test"]
    ) -> torch.Tensor:
        """Handle all the necessary logic for a single training / validation forward pass.

        Args:
            batch (Batch): A batch of data from a PriceAttributionDataset.
            step_type (Literal["train", "val", "test"]: Specifies which type of step we are taking.

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
        elif step_type == "test":
            precision = self.test_precision
            recall = self.test_recall
            f1 = self.test_f1
        else:
            raise NotImplementedError(
                "Unsupported step_type passed to PriceAttributor._step"
            )
        real_edges, fake_edges = get_candidate_edges(
            batch, balanced=self.balanced_edge_sampling and step_type == "train"
        )
        loss = torch.tensor(0.0, device=self.device, requires_grad=step_type == "train")
        for edges, label in [(real_edges, 1.0), (fake_edges, 0.0)]:
            if edges.size(1) > 0:
                logits = self(
                    x=batch.x,
                    edge_index=batch.edge_index,
                    src=edges[0],
                    dst=edges[1],
                    cluster_assignment=batch.get("upc_clusters"),
                )
                targets = torch.full_like(logits, fill_value=label)
                loss = loss + F.binary_cross_entropy_with_logits(logits, targets)

                probs = torch.sigmoid(logits)
                precision.update(probs, targets)
                recall.update(probs, targets)
                f1.update(probs, targets)
        self.log(
            f"{step_type}/loss",
            loss,
            on_epoch=True,
            on_step=False,
            prog_bar=True,
            batch_size=batch.num_graphs,
        )
        return loss

    def on_train_epoch_end(self):
        precision = self.trn_precision.compute()
        recall = self.trn_recall.compute()
        f1 = self.trn_f1.compute()
        self.log("train/precision_mean", precision["mean"])
        self.log("train/precision_std", precision["std"])
        self.log("train/recall_mean", recall["mean"])
        self.log("train/recall_std", recall["std"])
        self.log("train/f1_mean", f1["mean"])
        self.log("train/f1_std", f1["std"])

    def on_validation_epoch_end(self):
        precision = self.val_precision.compute()
        recall = self.val_recall.compute()
        f1 = self.val_f1.compute()
        self.log("val/precision_mean", precision["mean"])
        self.log("val/precision_std", precision["std"])
        self.log("val/recall_mean", recall["mean"])
        self.log("val/recall_std", recall["std"])
        self.log("val/f1_mean", f1["mean"])
        self.log("val/f1_std", f1["std"])

    def on_test_epoch_end(self):
        precision = self.test_precision.compute()
        recall = self.test_recall.compute()
        f1 = self.test_f1.compute()
        self.log("test/precision_mean", precision["mean"])
        self.log("test/precision_std", precision["std"])
        self.log("test/recall_mean", recall["mean"])
        self.log("test/recall_std", recall["std"])
        self.log("test/f1_mean", f1["mean"])
        self.log("test/f1_std", f1["std"])

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = CosineAnnealingLR(
            optimizer, T_max=self.num_epochs, eta_min=self.lr / 10
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
        }
