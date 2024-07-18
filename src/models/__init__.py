""" Models """

# from models.base import BaseConfig, BaseLmModel
from models.base import BaseConfig, BaseLmModel
from models.rat import RatConfig, RatLmModel
from models.rst import RSTConfig, RSTLmModel

CONFIG_DICT = {
    "base": BaseConfig,
    "rat": RatConfig,
    "rst": RSTConfig,
}

MODEL_DICT = {
    "base": BaseLmModel,
    "rat": RatLmModel,
    "rst": RSTLmModel,
}
