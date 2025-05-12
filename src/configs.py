from pathlib import Path

from pydantic import BaseModel

from src.common.enums import Accelerator
from src.common.enums import EncoderType
from src.common.enums import LinkPredictorType


class ModelConfig(BaseModel):
    encoder_type: EncoderType
    encoder_settings: dict
    link_predictor_type: LinkPredictorType
    link_predictor_settings: dict


class TrainingConfig(BaseModel):
    run_name: str
    num_epochs: int
    batch_size: int = 1
    lr: float = 3e-4
    weight_decay: float = 1e-5
    num_workers: int = 0
    accelerator: Accelerator = Accelerator.CPU
    balanced_edge_sampling: bool = True
    model: ModelConfig
    dataset_dir: Path
    log_dir: Path = Path("logs")
    chkp_dir: Path = Path("chkp")
