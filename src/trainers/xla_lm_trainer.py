import torch
import torch.nn as nn
import torch.nn.functional as F

from trainers.base_xla_trainer import BaseXLATrainer
from utils.data_utils import DotDict
from  utils.training_utils import loss, ppl, acc, pcorr


class XLALMTrainer(BaseXLATrainer):

    def train_step(self, model, tokenizer, x, seg_ids):

        out = model(x, segment_ids=seg_ids)
        ignore_index = tokenizer.pad_token_id

        eos_index = None
        if model.ignore_segment_ids:
            eos_index = tokenizer.eos_token_id

        results = DotDict(
            lm_loss=loss(out, x, ignore_index, eos_index),
            lm_ppl=ppl(out, x, ignore_index, eos_index),
            lm_acc=acc(out, x, ignore_index, eos_index),
            lm_pcorr=pcorr(out, x, ignore_index, eos_index),
        )
        results.loss = results.lm_loss

        return results
