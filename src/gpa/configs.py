from __future__ import annotations

from pathlib import Path
from typing import Literal

from gpa.common.enums import Accelerator
from gpa.common.enums import ConnectionStrategy
from gpa.common.enums import EncoderType
from gpa.common.enums import LinkPredictorType
from pydantic import BaseModel
from pydantic import model_validator


class PretrainedModelSpec(BaseModel):
    trn_config_path: Path
    chkp_path: Path


class InitialConnectionConfig(BaseModel):
    method: Literal["heuristic", "seed_model"] | None = None
    heuristic: ConnectionStrategy | None = None
    seed_model: PretrainedModelSpec | None = None

    @model_validator(mode="after")
    def check_consistency(self) -> InitialConnectionConfig:
        if self.method is None:
            return self
        if self.method == "heuristic":
            if self.heuristic is None:
                raise ValueError("heuristic must be set if method='heuristic'")
        elif self.method == "seed_model":
            if self.seed_model is None:
                raise ValueError("Model spec must be provided if method='seed_model'")
        else:
            raise ValueError(
                "Invalid connection method specified. Must be 'heuristic' or 'seed_model' if provided"
            )
        return self


class LoggingConfig(BaseModel):
    use_wandb: bool = False
    project_name: str | None = None
    log_dir: Path = Path("logs")
    chkp_dir: Path = Path("chkp")


class ModelConfig(BaseModel):
    # Architecture
    encoder_type: EncoderType
    encoder_settings: dict = {}
    link_predictor_type: LinkPredictorType
    link_predictor_settings: dict = {}

    # Preprocessing
    use_visual_info: bool = False
    use_depth: bool = True
    use_spatially_invariant_coords: bool = False
    initial_connection: InitialConnectionConfig = InitialConnectionConfig()


class TrainingConfig(BaseModel):
    run_name: str
    model: ModelConfig
    logging: LoggingConfig
    num_epochs: int
    batch_size: int = 1
    num_workers: int = 0
    gamma: float = 0.0
    accelerator: Accelerator = Accelerator.CPU
    lr: float = 3e-4
    weight_decay: float = 1e-5
    balanced_edge_sampling: bool = True
    edge_deletion_only: bool = False
    dataset_dir: Path


class EvalConfig(PretrainedModelSpec):
    results_dir: Path
