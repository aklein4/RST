""" Models """

from models.base import BaseConfig, BaseLmModel
from models.idr import IDRConfig, IDRLmModel
from models.rat import RatConfig, RatLmModel

CONFIG_DICT = {
    "base": BaseConfig,
    "idr": IDRConfig,
    "rat": RatConfig,
}

MODEL_DICT = {
    "base": BaseLmModel,
    "idr": IDRLmModel,
    "rat": RatLmModel,
}
