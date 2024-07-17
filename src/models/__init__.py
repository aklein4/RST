""" Models """

# from models.base import BaseConfig, BaseLmModel
from models.og_base import BaseConfig, BaseLmModel
from models.rat import RatConfig, RatLmModel

CONFIG_DICT = {
    "base": BaseConfig,
    "rat": RatConfig,
}

MODEL_DICT = {
    "base": BaseLmModel,
    "rat": RatLmModel,
}
