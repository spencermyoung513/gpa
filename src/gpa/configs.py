from pathlib import Path

from gpa.common.enums import Accelerator
from gpa.common.enums import ConnectionStrategy
from gpa.common.enums import EncoderType
from gpa.common.enums import LinkPredictorType
from pydantic import BaseModel


class LoggingConfig(BaseModel):
    use_wandb: bool = False
    project_name: str | None = None
    log_dir: Path = Path("logs")
    chkp_dir: Path = Path("chkp")


class ModelConfig(BaseModel):
    use_visual_info: bool = False
    aggregate_by_upc: bool = False
    use_spatially_invariant_coords: bool = False
    initial_connection_strategy: ConnectionStrategy | None = None
    encoder_type: EncoderType
    encoder_settings: dict = {}
    link_predictor_type: LinkPredictorType
    link_predictor_settings: dict = {}


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
    dataset_dir: Path
