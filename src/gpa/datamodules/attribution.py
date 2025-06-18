from pathlib import Path

import lightning as L
from gpa.common.enums import ConnectionStrategy
from gpa.datasets.attribution import PriceAttributionDataset
from gpa.training.transforms import HeuristicallyConnectGraph
from gpa.training.transforms import MakeBoundingBoxTranslationInvariant
from gpa.training.transforms import MaskOutVisualInformation
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import Compose


class PriceAttributionDataModule(L.LightningDataModule):
    def __init__(
        self,
        data_dir: Path,
        batch_size: int = 1,
        num_workers: int = 0,
        use_visual_info: bool = False,
        use_spatially_invariant_coords: bool = False,
        initial_connection_strategy: ConnectionStrategy | None = None,
    ):
        """Initialize a `PriceAttributionDataModule`.

        Args:
            data_dir (Path): The directory where the dataset is stored.
            batch_size (int, optional): The batch size to use for dataloaders. Defaults to 1.
            num_workers (int, optional): The number of workers to use for dataloaders. Defaults to 0.
            use_visual_info (bool, optional): Whether/not to use visual information as part of initial node representations. Defaults to False.
            use_spatially_invariant_coords (bool, optional): Whether/not to use spatially invariant coordinates as part of initial node representations. Defaults to False.
            initial_connection_strategy (InitialConnectionStrategy | None, optional): If provided, the strategy to use for initially connecting product/price nodes within a graph before passing it through the model. Defaults to None (only nodes with the same UPC will be connected).
        """
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.use_visual_info = use_visual_info
        self.use_spatially_invariant_coords = use_spatially_invariant_coords
        self.initial_connection_strategy = initial_connection_strategy

    def setup(self, stage: str):
        train_transforms = self._get_train_transforms()
        val_transforms = self._get_val_transforms()
        test_transforms = self._get_test_transforms()
        self.train = PriceAttributionDataset(
            root=self.data_dir / "train",
            transform=train_transforms,
        )
        self.val = PriceAttributionDataset(
            root=self.data_dir / "val",
            transform=val_transforms,
        )
        self.test = PriceAttributionDataset(
            root=self.data_dir / "test",
            transform=test_transforms,
        )

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

    def test_dataloader(self):
        return DataLoader(
            dataset=self.test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
            shuffle=False,
        )

    def _get_train_transforms(self) -> Compose:
        transforms = []
        if self.use_spatially_invariant_coords:
            transforms.append(MakeBoundingBoxTranslationInvariant())
        if not self.use_visual_info:
            transforms.append(MaskOutVisualInformation())
        if self.initial_connection_strategy is not None:
            transforms.append(
                HeuristicallyConnectGraph(self.initial_connection_strategy)
            )
        return Compose(transforms)

    def _get_val_transforms(self) -> Compose:
        transforms = []
        if self.use_spatially_invariant_coords:
            transforms.append(MakeBoundingBoxTranslationInvariant())
        if not self.use_visual_info:
            transforms.append(MaskOutVisualInformation())
        if self.initial_connection_strategy is not None:
            transforms.append(
                HeuristicallyConnectGraph(self.initial_connection_strategy)
            )
        return Compose(transforms)

    def _get_test_transforms(self) -> Compose:
        transforms = []
        if self.use_spatially_invariant_coords:
            transforms.append(MakeBoundingBoxTranslationInvariant())
        if not self.use_visual_info:
            transforms.append(MaskOutVisualInformation())
        if self.initial_connection_strategy is not None:
            transforms.append(
                HeuristicallyConnectGraph(self.initial_connection_strategy)
            )
        return Compose(transforms)
