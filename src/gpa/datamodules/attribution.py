from pathlib import Path

import lightning as L
from gpa.datasets.attribution import PriceAttributionDataset
from gpa.training.transforms import SampleRandomSubgraph
from torch_geometric.loader import DataLoader


class PriceAttributionDataModule(L.LightningDataModule):
    def __init__(
        self,
        data_dir: Path,
        batch_size: int = 1,
        num_workers: int = 0,
        a: float = 1.0,
        b: float = 1.0,
    ):
        """Initialize a `PriceAttributionDataModule`.

        Args:
            data_dir (Path): The directory where the dataset is stored.
            batch_size (int, optional): The batch size to use for dataloaders. Defaults to 1.
            num_workers (int, optional): The number of workers to use for dataloaders. Defaults to 0.
            a (float, optional): The alpha parameter of the beta distribution we sample p from (for forming subgraphs). Defaults to 1.0.
            b (float, optional): The beta parameter of the beta distribution we sample p from (for forming subgraphs). Defaults to 1.0.
        """
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.a = a
        self.b = b

    def setup(self, stage: str):
        self.train = PriceAttributionDataset(
            root=self.data_dir / "train",
            # This transform feeds the model a new subgraph each time (at various stages of completion).
            transform=SampleRandomSubgraph(self.a, self.b),
        )
        self.val = PriceAttributionDataset(
            root=self.data_dir / "val",
            # So our validation loop can mirror the training one, we once again apply random subsampling.
            transform=SampleRandomSubgraph(self.a, self.b),
        )
        # No transform (so we can assess the efficacy of our iterative inference scheme).
        self.inference = PriceAttributionDataset(root=self.data_dir / "val")

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
            shuffle=False,
        )

    def inference_dataloader(self):
        """Return a dataloader to be used for inference."""
        return DataLoader(
            dataset=self.inference,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
            shuffle=False,
        )
