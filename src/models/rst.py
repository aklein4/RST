from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.base import (
    BaseConfig, BaseTransformer, BaseLmModel,
    BaseAttention, BaseMLP
)


class RSTConfig(BaseConfig):

    model_type = 'RST'

    def __init__(
        self,
        selection_size: int = 4,
        delta_rank: int = 1,
        *args,
        **kwargs,
    ):
        
        self.selection_size = selection_size
        self.delta_rank = delta_rank

        super().__init__(*args, **kwargs)


class SplitNorm(nn.Module):

    def __init__(self, hidden_size, selection_size, eps):
        super().__init__()

        self.hidden_size = hidden_size
        self.selection_size = selection_size
        self.residual_size = hidden_size - selection_size

        self.selection_norm = nn.LayerNorm(selection_size, eps=eps)
        self.residual_norm = nn.LayerNorm(self.residual_size, eps=eps)


    def forward(self, hidden_states):
        selection_states, residual_states = hidden_states.split(
            [self.selection_size, self.residual_size],
            dim=-1
        )

        selection_states = self.selection_norm(selection_states)
        residual_states = self.residual_norm(residual_states)

        return torch.cat([selection_states, residual_states], dim=-1)


class SSMConnection(nn.Module):

    def special_init_weights(self, config):

        # TODO: is this is best way?
        self.A.data.normal_(0.0, config.initializer_range)

        self.delta_up.weight.data.zero_()

        # TODO: this is hardcoded from MAMBA
        t = 0.001 + (
            torch.rand_like(self.delta_up.bias.data) * (0.1 - 0.001)
        )
        self.delta_up.bias.data = torch.log(t.exp() - 1)


    def __init__(self, config, y_size):
        super().__init__()

        self.hidden_size = config.hidden_size
        self.selection_size = config.selection_size
        self.residual_size = self.hidden_size - self.selection_size
        self.y_size = y_size

        # decay factor
        self.A = nn.Parameter(-torch.ones(1, 1, self.selection_size))

        # project y into the stream
        self.W = nn.Linear(self.y_size, self.hidden_size, bias=False)
        
        # calculate delta from y and hidden states
        self.delta_norm = SplitNorm(self.hidden_size, self.selection_size, config.layer_norm_eps)
        self.delta_down = nn.Linear(self.hidden_size + self.y_size, config.delta_rank, bias=False)
        self.delta_up = nn.Linear(config.delta_rank, self.selection_size, bias=True)


    def forward(self, hidden_states, y):

        # split components
        selection_states, residual_states = hidden_states.split(
            [self.selection_size, self.residual_size],
            dim=-1
        )
        selection_y, residual_y = self.W(y).split(
            [self.selection_size, self.residual_size],
            dim=-1
        )

        # get delta
        delta = self.delta_up(
            self.delta_down(
                torch.cat(
                    [
                        self.delta_norm(hidden_states),
                        y
                    ],
                    dim=-1
                )
            )
        )
        delta = F.softplus(delta)

        # calculate SSM matrices
        A_neg = -F.softplus(self.A)
        A_bar = torch.exp(delta * A_neg)
        B_bar = (A_bar - 1) / A_neg

        selection_states = A_bar * selection_states + B_bar * selection_y
        residual_states = residual_states + residual_y

        return torch.cat([selection_states, residual_states], dim=-1)


class RSTAttention(BaseAttention):

    def init_o_proj(self, config):
        pass

    def get_o(self, hidden_states):
        return hidden_states


class RSTMLP(BaseMLP):

    def init_mlp_output(self, config):
        pass

    def get_mlp_output(self, hidden_states):
        return hidden_states


class RSTLayer(nn.Module):

    def special_init_weights(self, config):
        if config.identity_init:
            raise ValueError("Identity init not supported for SSMConnection!")

        self.attn_connection.special_init_weights(config)
        self.mlp_connection.special_init_weights(config)

    def post_step(self):
        pass


    def __init__(self, config, layer_idx: int):
        super().__init__()

        self.hidden_size = config.hidden_size

        self.attn = RSTAttention(config, layer_idx)
        self.mlp = RSTMLP(config)

        h = config.hidden_size
        eps = config.layer_norm_eps

        self.attn_norm = SplitNorm(h, config.selection_size, eps)
        self.mlp_norm = SplitNorm(h, config.selection_size, eps)

        self.attn_connection = SSMConnection(config, self.hidden_size)
        self.mlp_connection = SSMConnection(config, config.mlp_size)


    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value = None,
    ):

        # Self Attention
        attn_out = self.attn(
            self.attn_norm(hidden_states),
            position_ids,
            attention_mask,
            past_key_value=past_key_value
        )
        hidden_states = self.attn_connection(hidden_states, attn_out)

        # GLU MLP
        mlp_out = self.mlp(
            self.mlp_norm(hidden_states)
        )
        hidden_states = self.mlp_connection(hidden_states, mlp_out)

        return hidden_states


class RSTTransformer(BaseTransformer):

    layer_type = RSTLayer


    def special_init_weights(self, config: BaseConfig):
        super().special_init_weights(config)

        self.vocab_embs.weight.data[:, :config.selection_size].zero_()


    def get_extras(self, config):
        self.norm = SplitNorm(config.hidden_size, config.selection_size, config.layer_norm_eps)


class RSTLmModel(BaseLmModel):

    transformer_type = RSTTransformer
