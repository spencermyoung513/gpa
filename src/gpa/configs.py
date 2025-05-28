from pathlib import Path

from gpa.common.enums import Accelerator
from gpa.common.enums import EncoderType
from gpa.common.enums import LinkPredictorType
from pydantic import BaseModel
from pydantic import Field


class ModelConfig(BaseModel):
    encoder_type: EncoderType
    encoder_settings: dict
    link_predictor_type: LinkPredictorType
    link_predictor_settings: dict


class TrainingConfig(BaseModel):
    run_name: str
    num_epochs: int
    batch_size: int = 1
    num_workers: int = 0
    accelerator: Accelerator = Accelerator.CPU
    lr: float = 3e-4
    weight_decay: float = 1e-5
    a: float = Field(default=1.0, gt=0.0)
    b: float = Field(default=1.0, gt=0.0)
    balanced_edge_sampling: bool = True
    model: ModelConfig
    dataset_dir: Path
    log_dir: Path = Path("logs")
    chkp_dir: Path = Path("chkp")
