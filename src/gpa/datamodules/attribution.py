from pathlib import Path
from typing import Literal

import lightning as L
from gpa.datasets.attribution import PriceAttributionDataset
from gpa.training.transforms import HeuristicallyConnectGraph
from gpa.training.transforms import MakeBoundingBoxTranslationInvariant
from gpa.training.transforms import MaskOutVisualInformation
from gpa.training.transforms import RemoveUPCClusters
from gpa.training.transforms import SampleRandomSubgraph
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import Compose


class PriceAttributionDataModule(L.LightningDataModule):
    def __init__(
        self,
        data_dir: Path,
        batch_size: int = 1,
        num_workers: int = 0,
        a: float | None = None,
        b: float | None = None,
        use_visual_info: bool = False,
        aggregate_by_upc: bool = False,
        use_spatially_invariant_coords: bool = False,
        initial_connection_scheme: Literal["nearest", "nearest_below"] | None = None,
    ):
        """Initialize a `PriceAttributionDataModule`.

        Args:
            data_dir (Path): The directory where the dataset is stored.
            batch_size (int, optional): The batch size to use for dataloaders. Defaults to 1.
            num_workers (int, optional): The number of workers to use for dataloaders. Defaults to 0.
            a (float, optional): If specified, the alpha parameter of the beta distribution from which we sample our edge dropout probability (for forming subgraphs). Defaults to None (no edge dropout).
            b (float, optional): If specified, the beta parameter of the beta distribution from which we sample our edge dropout probability (for forming subgraphs). Defaults to None (no edge dropout).
            use_visual_info (bool, optional): Whether/not to use visual information as part of initial node representations. Defaults to False.
            aggregate_by_upc (bool, optional): Whether/not to aggregate node embeddings by UPC after encoding. Defaults to False.
            use_spatially_invariant_coords (bool, optional): Whether/not to use spatially invariant coordinates as part of initial node representations. Defaults to False.
            initial_connection_scheme (Literal["nearest", "nearest_below"] | None, optional): If provided, the scheme to use for initially connecting product/price nodes within a graph before passing it through the model. Defaults to None (only nodes with the same UPC will be connected).
        """
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.a = a
        self.b = b
        self.use_visual_info = use_visual_info
        self.aggregate_by_upc = aggregate_by_upc
        self.use_spatially_invariant_coords = use_spatially_invariant_coords
        self.initial_connection_scheme = initial_connection_scheme

    def setup(self, stage: str):
        train_transforms = self._get_train_transforms()
        val_transforms = self._get_val_transforms()
        inference_transforms = self._get_inference_transforms()
        self.train = PriceAttributionDataset(
            root=self.data_dir / "train",
            transform=train_transforms,
        )
        self.val = PriceAttributionDataset(
            root=self.data_dir / "val",
            transform=val_transforms,
        )
        self.inference = PriceAttributionDataset(
            root=self.data_dir / "val",
            transform=inference_transforms,
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

    def inference_dataloader(self):
        return DataLoader(
            dataset=self.inference,
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
        if not self.aggregate_by_upc:
            transforms.append(RemoveUPCClusters())
        if self.a is not None and self.b is not None:
            transforms.append(SampleRandomSubgraph(self.a, self.b))
        if self.initial_connection_scheme is not None:
            transforms.append(HeuristicallyConnectGraph(self.initial_connection_scheme))
        return Compose(transforms)

    def _get_val_transforms(self) -> Compose:
        transforms = []
        if self.use_spatially_invariant_coords:
            transforms.append(MakeBoundingBoxTranslationInvariant())
        if not self.use_visual_info:
            transforms.append(MaskOutVisualInformation())
        if not self.aggregate_by_upc:
            transforms.append(RemoveUPCClusters())
        if self.a is not None and self.b is not None:
            # We want validation to mirror the same setup as training,
            # so we also perform subgraph sampling.
            transforms.append(SampleRandomSubgraph(self.a, self.b))
        if self.initial_connection_scheme is not None:
            transforms.append(HeuristicallyConnectGraph(self.initial_connection_scheme))
        return Compose(transforms)

    def _get_inference_transforms(self) -> Compose:
        transforms = []
        if self.use_spatially_invariant_coords:
            transforms.append(MakeBoundingBoxTranslationInvariant())
        if not self.use_visual_info:
            transforms.append(MaskOutVisualInformation())
        if not self.aggregate_by_upc:
            transforms.append(RemoveUPCClusters())
        if self.initial_connection_scheme is not None:
            transforms.append(HeuristicallyConnectGraph(self.initial_connection_scheme))
        return Compose(transforms)
