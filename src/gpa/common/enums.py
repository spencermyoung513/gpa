from enum import Enum


class EncoderType(Enum):
    TRANSFORMER = "transformer"
    GAT = "gat"
    IDENTITY = "identity"


class LinkPredictorType(Enum):
    MLP = "mlp"
    INNER_PRODUCT = "inner_product"


class Accelerator(Enum):
    CPU = "cpu"
    GPU = "gpu"
    MPS = "mps"


class ConnectionStrategy(Enum):
    NEAREST = "nearest"
    NEAREST_BELOW = "nearest_below"
    NEAREST_BELOW_PER_GROUP = "nearest_below_per_group"
