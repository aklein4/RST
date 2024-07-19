import torch

import numpy as np

from transformers import AutoTokenizer

from models.rat import RatConfig, RatLmModel
from models.base import BaseConfig, BaseLmModel
from utils.config_utils import load_model_config
import utils.constants as constants


MODEL_CONFIG = 'med-rat'
BASE_CONFIG = 'med-base'

COMPARE = True


def main():

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(constants.GPT2_TOKENIZER, resume_download=None)
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    if COMPARE:

        print("loading model...")
        config = load_model_config(MODEL_CONFIG, tokenizer)
        model = RatLmModel(RatConfig(**config), fast_start=True)

        print("loading base...")
        base_config = load_model_config(BASE_CONFIG, tokenizer)
        base_model = BaseLmModel(BaseConfig(**base_config), fast_start=True)
    
        n = np.sum([p.numel() for p in model.parameters()])
        n_base = np.sum([p.numel() for p in base_model.parameters()])

        print(f"Rat: {n:_} parameters")
        print(f"Base: {n_base:_} parameters")
        print(f"Diff: {n - n_base:_}")

        return

    x = tokenizer(["Hello, my dog is cute", "His dog is cute too", "All dogs are cute"], return_tensors="pt", padding="max_length", max_length=16).input_ids

    print("loading model...")
    config = load_model_config(MODEL_CONFIG, tokenizer)
    model = RatLmModel(RatConfig(**config))
    model.post_step()

    out = model(x)

    # print(out)
    print(out.shape)


if __name__ == '__main__':
    main()
