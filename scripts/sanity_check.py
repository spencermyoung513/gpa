from pathlib import Path

import torch
from gpa.common.helpers import get_candidate_edges
from gpa.datamodules.attribution import PriceAttributionDataModule
from torchmetrics import BootStrapper
from torchmetrics.classification import BinaryF1Score
from torchmetrics.classification import BinaryPrecision
from torchmetrics.classification import BinaryRecall
from tqdm import tqdm


if __name__ == "__main__":
    datamodule = PriceAttributionDataModule(data_dir=Path("data/price-graphs-ii"))
    datamodule.setup("")
    precision = BootStrapper(BinaryPrecision())
    recall = BootStrapper(BinaryRecall())
    f1 = BootStrapper(BinaryF1Score())

    for batch in tqdm(datamodule.test_dataloader()):
        real_edges, fake_edges = get_candidate_edges(batch, balanced=False)
        for edges, label in [(real_edges, 1.0), (fake_edges, 0.0)]:
            if edges.size(1) > 0:
                probs = torch.ones(edges.shape[1])
                targets = torch.full_like(probs, fill_value=label)
                precision.update(probs, targets)
                recall.update(probs, targets)
                f1.update(probs, targets)

    print(f"Precision: {precision.compute()['mean'].item():.4f}")
    print(f"Recall: {recall.compute()['mean'].item():.4f}")
    print(f"F1: {f1.compute()['mean'].item():.4f}")
