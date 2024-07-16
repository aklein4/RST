from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.base import (
    BaseConfig, BaseTransformer, BaseLmModel,
    BaseAttention, BaseMLP
)


class IDRConfig(BaseConfig):

    model_type = 'idr'

    def __init__(
        self,
        register_size: int = 128,
        *args,
        **kwargs,
    ):
        
        self.register_size = register_size

        super().__init__(*args, **kwargs)


class IDRInputNorm(nn.Module):

    def __init__(self, hidden_size, register_size, eps):
        super().__init__()
        assert register_size < hidden_size

        self.hidden_size = hidden_size
        self.register_size = register_size

        self.register_norm = nn.LayerNorm(register_size, eps=eps)
        self.skip_norm = nn.LayerNorm(hidden_size, eps=eps)


    def forward(self, hidden_states):
        register_states, skip_states = hidden_states.split(
            [
                self.register_size,
                self.hidden_size - self.register_size
            ],
            dim=-1
        )

        register_states = self.register_norm(register_states)
        skip_states = self.skip_norm(skip_states)

        return torch.cat([register_states, skip_states], dim=-1)


class IDRSkipNorm(nn.Module):

    def __init__(self, hidden_size, register_size, eps):
        super().__init__()
        assert register_size < hidden_size

        self.hidden_size = hidden_size
        self.register_size = register_size

        self.register_norm = nn.LayerNorm(register_size, eps=eps)


    def forward(self, hidden_states):
        register_states, skip_states = hidden_states.split(
            [
                self.register_size,
                self.hidden_size - self.register_size
            ],
            dim=-1
        )

        register_states = self.register_norm(register_states)

        return torch.cat([register_states, skip_states], dim=-1)


class IDRLayer(nn.Module):
    def __init__(self, config: IDRConfig, layer_idx: int):
        super().__init__()

        self.hidden_size = config.hidden_size

        self.attn = BaseAttention(config, layer_idx)
        self.mlp = BaseMLP(config)

        h = config.hidden_size
        r = config.register_size
        eps = config.layer_norm_eps

        self.attn_input_norm = IDRInputNorm(h, r, eps)
        self.mlp_input_norm = IDRSkipNorm(h, r, eps)

        self.attn_skip_norm = IDRSkipNorm(h, r, eps)
        self.mlp_skip_norm = IDRSkipNorm(h, r, eps)


    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value = None,
    ):

        # Self Attention
        attn_out = self.attn(
            self.attn_input_norm(hidden_states),
            attention_mask,
            past_key_value=past_key_value
        )
        hidden_states = self.attn_skip_norm(hidden_states) + attn_out

        # GLU MLP
        mlp_out = self.mlp(
            self.mlp_input_norm(hidden_states)
        )
        hidden_states = self.mlp_skip_norm(hidden_states) + mlp_out

        return hidden_states


class IDRTransformer(BaseTransformer):
    layer_type = IDRLayer

    def get_norm(self, config):
        return IDRInputNorm(config.hidden_size, config.register_size, config.layer_norm_eps)


class IDRLmModel(BaseLmModel):
    transformer_type = IDRTransformer
