from pathlib import Path

import lightning as L
from torch_geometric.loader import DataLoader

from gpa.datasets.attribution import PriceAttributionDataset


class PriceAttributionDataModule(L.LightningDataModule):
    def __init__(self, data_dir: Path, batch_size: int = 1, num_workers: int = 0):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage: str):
        self.train = PriceAttributionDataset(root=self.data_dir / "train")
        self.val = PriceAttributionDataset(root=self.data_dir / "val")

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )
