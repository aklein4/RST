""" Models """

from models.base import BaseConfig, BaseLmModel
from models.idr import IDRConfig, IDRLmModel

CONFIG_DICT = {
    "base": BaseConfig,
    "idr": IDRConfig,
}

MODEL_DICT = {
    "base": BaseLmModel,
    "idr": IDRLmModel,
}
