from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.base import (
    BaseConfig, BaseTransformer, BaseLmModel,
    BaseAttention, BaseMLP
)
from utils.model_utils import fast_checkpoint


class RatConfig(BaseConfig):

    model_type = 'rat'

    def __init__(
        self,
        residual_channels: int = 4,
        residual_heads: int = 16,
        dot_norm_eps: float = 1e-5,
        *args,
        **kwargs,
    ):
        
        self.residual_channels = residual_channels
        self.residual_heads = residual_heads
        self.dot_norm_eps = dot_norm_eps

        super().__init__(*args, **kwargs)


class RatRead(nn.Module):


    def special_init_weights(self, config):
        self.conv.weight.data.normal_()
        self.post_step()


    def post_step(self):
        self.conv.weight.data[:] = (
            self.conv.weight.data / 
            (
                self.conv.weight.data.norm(p=2, dim=1, keepdim=True) +
                self.rat_norm_eps
            )
        ).detach()


    def __init__(self, config: RatConfig, num_outputs):
        super().__init__()

        self.hidden_size = config.hidden_size
        self.residual_channels = config.residual_channels
        self.residual_size = self.hidden_size * self.residual_channels
        
        self.residual_heads = config.residual_heads
        assert self.hidden_size % self.residual_heads == 0

        self.num_outputs = num_outputs
        self.output_size = self.hidden_size * self.num_outputs

        self.conv = nn.Conv1d(
            self.residual_size, self.output_size,
            kernel_size=1, bias=False,
            groups=self.hidden_size
        )

        self.norm = nn.GroupNorm(
            self.num_outputs,
            self.output_size,
            eps=config.layer_norm_eps
        )


    def compute(self, hidden_states):
        bs, l, _ = hidden_states.shape

        # [B, L, HN]
        hidden_states = hidden_states.view(bs*l, self.residual_size, 1)
        hidden_states = self.conv(hidden_states)
        hidden_states = hidden_states.view(bs, l, self.output_size)

        # [B, L, N, H]
        hidden_states = (
            hidden_states
            .view(bs, l, self.hidden_size, self.num_outputs)
            .permute(0, 1, 3, 2)
            .reshape(bs, l, self.output_size)
        )

        return self.norm(hidden_states)
    

    def forward(self, hidden_states):
        return fast_checkpoint(self.compute, hidden_states)


class RatWrite(nn.Module):

    def special_init_weights(self, config):

        # TODO: is this is best way?
        self.A.data.normal_(0.0, config.initializer_range)

        self.delta_up.weight.data.zero_()

        # TODO: this is hardcoded from MAMBA
        t = 0.001 + (
            torch.rand_like(self.delta_up.bias.data) * (0.1 - 0.001)
        )

        # residual gets t=1
        t[..., self.half_size:].fill_(1.0)

        self.delta_up.bias.data = torch.log(t.exp() - 1)


    def __init__(self, config, y_size):
        super().__init__()

        self.hidden_size = config.hidden_size
        self.residual_channels = config.residual_channels
        self.residual_heads = config.residual_heads
        
        self.head_size = self.hidden_size // self.residual_heads
        assert self.hidden_size % self.residual_heads == 0
        
        self.A_size = (1 - self.residual_channels) * self.residual_heads
        self.B_size = self.residual_channels * self.residual_heads

        self.y_size = y_size
        self.ssm_epsilon = config.ssm_epsilon

        # projections for y
        self.Wo_Wd = nn.Linear(self.y_size, self.hidden_size + config.delta_rank, bias=False)
        self.WA_WB = nn.Linear(self.y_size, self.A_size + self.B_size, bias=True)
        
        # up projection for delta
        self.delta_up = nn.Linear(config.delta_rank, self.hidden_size, bias=True)


    def compute(self, hidden_states, o_d, A_B):
        bs, l, _ = hidden_states.shape

        # split o_d
        out = o_d[:, :, :self.hidden_size]
        delta_low = o_d[:, :, self.hidden_size:]

        # split A_B
        A_logit = A_B[:, :, :self.A_size] # A_size
        B_raw = A_B[:, :, self.A_size:] # B_size
        B_proj = B_raw[:, :, :self.A_size] # A_size
        B_skip = B_raw[:, :, self.A_size:] # heads

        # get delta
        delta = self.delta_up(delta_low)
        delta = F.softplus(delta)

        # reshape for SSM
        out = out.view(bs, l, self.head_size, self.residual_heads, 1)
        delta = delta.
        A_raw = A_raw.view(bs, l, self.head_size, 1, self.residual_channels)
        B_raw = B_raw.view(bs, l, self.head_size, 1, self.residual_channels)


        # calculate SSM matrices
        A_neg = -F.softplus(A_raw)
        A_bar = torch.exp(delta[:, :, :self.half_size] * A_neg)
        B_bar = (A_bar - 1) / (A_neg - self.ssm_epsilon)

        # add residuals
        A_bar = torch.cat([A_bar, torch.ones_like(A_bar)], dim=-1)
        B_bar = torch.cat([B_bar, delta[:, :, self.half_size:]], dim=-1)

        return (
            A_bar * hidden_states +
            B_bar * y_out
        )


    def forward(self, hidden_states, y):

        # do major projections
        o_d = self.W(y)
        A_B = self.WA_WB(y)

        return fast_checkpoint(self.compute, hidden_states, o_d, A_B)


class RatAttention(BaseAttention):

    def _init_qkv_proj(self, config):
        self.qkv_proj = nn.Conv1d(
            3 * config.hidden_size, 3 * config.hidden_size,
            kernel_size=1, bias=config.use_qkv_bias,
            groups=3
        )
    

    def _get_qkv(self, hidden_states):
        bs, l, _ = hidden_states.shape

        hidden_states = hidden_states.view(bs*l, 3 * self.hidden_size, 1)
        hidden_states = self.qkv_proj(hidden_states)
        hidden_states = hidden_states.view(bs, l, 3 * self.hidden_size)

        return hidden_states.chunk(3, dim=-1)


class RatMLP(BaseMLP):

    def _init_mlp_input(self, config):
        self.in_proj = nn.Conv1d(
            2*config.hidden_size, 2*config.mlp_size,
            kernel_size=1, bias=False,
            groups=2
        )

    
    def _get_mlp_input(self, hidden_states):
        bs, l, _ = hidden_states.shape

        hidden_states = hidden_states.view(bs*l, 2 * self.hidden_size, 1)
        hidden_states = self.in_proj(hidden_states)
        hidden_states = hidden_states.view(bs, l, 2 * self.mlp_size)

        return hidden_states.chunk(2, dim=-1)


class RatLayer(nn.Module):

    # @ torch.no_grad()()
    def special_init_weights(self, config: BaseConfig):
        if config.identity_init:
            self.attn.o_proj.weight.data.zero_()
            self.mlp.down_proj.weight.data.zero_()

        self.attn_input.special_init_weights(config)
        self.mlp_input.special_init_weights(config)

        self.attn_output.special_init_weights(config)
        self.mlp_output.special_init_weights(config)


    # @ torch.no_grad()()
    def post_step(self):
        
        self.attn_input.post_step()
        self.mlp_input.post_step()

        self.attn_output.post_step()
        self.mlp_output.post_step()


    def __init__(self, config: BaseConfig, layer_idx: int):
        super().__init__()

        self.hidden_size = config.hidden_size

        self.attn = RatAttention(config, layer_idx)
        self.mlp = RatMLP(config)

        self.attn_input = RatInput(config, 3)
        self.mlp_input = RatInput(config, 2)

        self.attn_output = RatOutput(config)
        self.mlp_output = RatOutput(config)


    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value=None,
    ):

        # Self Attention
        attn_out = self.attn(
            self.attn_input(hidden_states),
            position_ids,
            attention_mask,
            past_key_value=past_key_value
        )
        hidden_states = hidden_states + self.attn_output(attn_out)

        # GLU MLP
        mlp_out = self.mlp(
            self.mlp_input(hidden_states)
        )
        hidden_states = hidden_states + self.mlp_output(mlp_out)

        return hidden_states


class RatTransformer(BaseTransformer):
    layer_type = RatLayer

    def get_norm(self, config):
        self.proj_in = RatOutput(config)
        self.pos_in = RatOutput(config)
        self.norm = RatInput(config, 1)


    # @ torch.no_grad()()
    def special_init_weights(self, config):
        super().special_init_weights(config)

        self.proj_in.special_init_weights(config)
        self.pos_in.special_init_weights(config)
        self.norm.special_init_weights(config)
    

    # @ torch.no_grad()()
    def post_step(self):
        super().post_step()

        self.proj_in.post_step()
        self.pos_in.post_step()
        self.norm.post_step()

    
    def get_hidden_states(
        self,
        input_ids: torch.LongTensor,
        position_ids: torch.LongTensor
    ) -> torch.Tensor:
        tokens = self.vocab_embs(input_ids)
        pos = self.pos_embs(position_ids)

        return (
            self.proj_in(tokens) +
            self.pos_in(pos)
        )


class RatLmModel(BaseLmModel):

    transformer_type = RatTransformer

    requires_barrier = True