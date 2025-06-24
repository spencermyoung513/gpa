from pathlib import Path

import lightning as L
import yaml
from gpa.configs import InitialConnectionConfig
from gpa.configs import TrainingConfig
from gpa.datasets.attribution import PriceAttributionDataset
from gpa.training.transforms import ConnectGraphWithSeedModel
from gpa.training.transforms import FilterExtraneousPriceTags
from gpa.training.transforms import HeuristicallyConnectGraph
from gpa.training.transforms import MakeBoundingBoxTranslationInvariant
from gpa.training.transforms import MaskOutVisualInformation
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import BaseTransform
from torch_geometric.transforms import Compose


class PriceAttributionDataModule(L.LightningDataModule):
    def __init__(
        self,
        data_dir: Path,
        batch_size: int = 1,
        num_workers: int = 0,
        use_visual_info: bool = False,
        use_spatially_invariant_coords: bool = False,
        initial_connection_config: InitialConnectionConfig = InitialConnectionConfig(),
    ):
        """Initialize a `PriceAttributionDataModule`.

        Args:
            data_dir (Path): The directory where the dataset is stored.
            batch_size (int, optional): The batch size to use for dataloaders. Defaults to 1.
            num_workers (int, optional): The number of workers to use for dataloaders. Defaults to 0.
            use_visual_info (bool, optional): Whether/not to use visual information as part of initial node representations. Defaults to False.
            use_spatially_invariant_coords (bool, optional): Whether/not to use spatially invariant coordinates as part of initial node representations. Defaults to False.
            initial_connection_config (InitialConnectionConfig, optional): Configuration for how the graph should be initially connected (in addition to its sparse, same-UPC-only connections) before being passed to the model.
        """
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.use_visual_info = use_visual_info
        self.use_spatially_invariant_coords = use_spatially_invariant_coords
        self.initial_connection_config = initial_connection_config

    def setup(self, stage: str):
        transform = self._get_transform(
            initial_connection_config=self.initial_connection_config,
            use_spatially_invariant_coords=self.use_spatially_invariant_coords,
            use_visual_info=self.use_visual_info,
        )
        self.train = PriceAttributionDataset(
            root=self.data_dir / "train",
            transform=transform,
        )
        self.val = PriceAttributionDataset(
            root=self.data_dir / "val",
            transform=transform,
        )
        self.test = PriceAttributionDataset(
            root=self.data_dir / "test",
            transform=transform,
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

    @staticmethod
    def _get_transform(
        initial_connection_config: InitialConnectionConfig,
        use_spatially_invariant_coords: bool = False,
        use_visual_info: bool = False,
    ) -> BaseTransform:
        edge_modifiers = PriceAttributionDataModule._get_edge_modifiers(
            initial_connection_config=initial_connection_config
        )
        node_modifiers = PriceAttributionDataModule._get_node_modifiers(
            use_spatially_invariant_coords=use_spatially_invariant_coords,
            use_visual_info=use_visual_info,
        )
        return Compose([FilterExtraneousPriceTags(), edge_modifiers, *node_modifiers])

    @staticmethod
    def _get_edge_modifiers(
        initial_connection_config: InitialConnectionConfig,
    ) -> BaseTransform:
        if initial_connection_config.method == "heuristic":
            heuristic = initial_connection_config.heuristic
            assert heuristic is not None
            return HeuristicallyConnectGraph(heuristic)

        if initial_connection_config.method == "seed_model":
            seed_model_spec = initial_connection_config.seed_model
            assert seed_model_spec is not None
            with open(seed_model_spec.trn_config_path, "r") as f:
                seed_model_cfg = TrainingConfig(**yaml.safe_load(f))
            seed_node_modifiers = PriceAttributionDataModule._get_node_modifiers(
                use_spatially_invariant_coords=seed_model_cfg.model.use_spatially_invariant_coords,
                use_visual_info=seed_model_cfg.model.use_visual_info,
            )
            seed_transforms = [*seed_node_modifiers]
            assert seed_model_cfg.model.initial_connection.method != "seed_model", (
                "Recursive seed models not permitted"
            )
            if seed_model_cfg.model.initial_connection.method == "heuristic":
                heuristic = seed_model_cfg.model.initial_connection.heuristic
                assert heuristic is not None
                seed_transforms.append(HeuristicallyConnectGraph(heuristic))
            return ConnectGraphWithSeedModel(
                seed_model_chkp_path=seed_model_spec.chkp_path,
                seed_model_transform=Compose(seed_transforms),
            )

        elif initial_connection_config.method is not None:
            raise NotImplementedError("Initial connection method not suported")

    @staticmethod
    def _get_node_modifiers(
        use_spatially_invariant_coords: bool = False,
        use_visual_info: bool = False,
    ) -> list[BaseTransform]:
        transforms = []
        if use_spatially_invariant_coords:
            transforms.append(MakeBoundingBoxTranslationInvariant())
        if not use_visual_info:
            transforms.append(MaskOutVisualInformation())
        return transforms
