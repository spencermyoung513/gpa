from pathlib import Path
from typing import Literal

from gpa.common.enums import Accelerator
from gpa.common.enums import EncoderType
from gpa.common.enums import LinkPredictorType
from pydantic import BaseModel
from pydantic import Field
from pydantic import model_validator


class LoggingConfig(BaseModel):
    use_wandb: bool = False
    project_name: str | None = None
    log_dir: Path = Path("logs")
    chkp_dir: Path = Path("chkp")


class ModelConfig(BaseModel):
    use_visual_info: bool = False
    aggregate_by_upc: bool = False
    use_spatially_invariant_coords: bool = False
    initial_connection_scheme: Literal["nearest", "nearest_below"] | None = None
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
    accelerator: Accelerator = Accelerator.CPU
    lr: float = 3e-4
    weight_decay: float = 1e-5
    a: float | None = Field(default=None, gt=0.0)
    b: float | None = Field(default=None, gt=0.0)
    balanced_edge_sampling: bool = True
    dataset_dir: Path

    @model_validator(mode="after")
    def ensure_a_and_b_are_both_specified_or_neither_is(self):
        a_not_b = self.a is not None and self.b is None
        b_not_a = self.b is not None and self.a is None
        if a_not_b or b_not_a:
            raise ValueError("`a` and `b` must both be specified or both be None.")
        return self

    @model_validator(mode="after")
    def ensure_heuristic_and_subgraph_sampling_are_mutually_exclusive(self):
        if (
            self.a is not None
            and self.b is not None
            and self.model.initial_connection_scheme is not None
        ):
            raise ValueError(
                "Random subgraph sampling does not play well with initial heuristic connection. Please specify either (`a`, `b`) or `initial_connection_scheme`, but not both."
            )
        return self
