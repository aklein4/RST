import torch

from transformers import AutoTokenizer

from models.base import BaseConfig, BaseLmModel
from models.og_base import BaseConfig as OGBaseConfig, BaseLmModel as OGBaseLmModel
from utils.config_utils import load_model_config
import utils.constants as constants


MODEL_CONFIG = 'test-base'


def main():


    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(constants.GPT2_TOKENIZER, resume_download=None)
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    
    x = tokenizer(["Hello, my dog is cute", "His dog is cute too", "All dogs are cute"], return_tensors="pt", padding="max_length", max_length=16).input_ids
    seg_ids = torch.randint_like(x, 4)

    print("loading model...")
    config = load_model_config(MODEL_CONFIG, tokenizer)
    model = BaseLmModel(BaseConfig(**config))
    
    og_model = OGBaseLmModel(OGBaseConfig(**config))
    og_model.load_state_dict(model.state_dict(), strict=True)

    out = model(x, segment_ids=seg_ids)
    out_og = og_model(x, segment_ids=seg_ids)

    # print(out)
    print((out - out_og).abs().max().item())


if __name__ == '__main__':
    main()
