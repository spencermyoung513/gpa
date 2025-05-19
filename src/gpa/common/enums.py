from enum import Enum


class EncoderType(Enum):
    TRANSFORMER = "transformer"
    GAT = "gat"


class LinkPredictorType(Enum):
    MLP = "mlp"
    INNER_PRODUCT = "inner_product"


class Accelerator(Enum):
    CPU = "cpu"
    GPU = "gpu"
    MPS = "mps"
