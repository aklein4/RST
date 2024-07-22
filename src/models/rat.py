from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import torch_xla.core.xla_model as xm
except:
    pass

import numpy as np

from models.base import (
    BaseConfig, BaseTransformer, BaseLmModel,
    BaseAttention, BaseMLP
)
from utils.model_utils import _extract_tensors_from_list
import utils.constants as constants


class RatConfig(BaseConfig):

    model_type = 'rat'

    def __init__(
        self,
        residual_channels: int = 4,
        normalizer_eps: float = 1e-5,
        bootstrap_debug: bool = False,
        *args,
        **kwargs,
    ):
        
        self.residual_channels = residual_channels
        self.normalizer_eps = normalizer_eps
        self.bootstrap_debug = bootstrap_debug

        super().__init__(*args, **kwargs)


class Tracker(nn.Module):

    def __init__(self):
        super().__init__()
        self.tracked = None
        self.normalizer = None
    
    def forward(self, x, normalizer):
        return track_fn.apply(x, normalizer, self)


    def get(self):
        return self.tracked

    def update(self, x):
        self.tracked = x.detach()


    def get_normalizer(self):
        return self.normalizer
    
    def update_normalizer(self, x):
        self.normalizer = x.detach()


    def clear(self):
        self.tracked = None
        self.normalizer = None


class track_fn(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, normalizer, tracker):
        ctx.tracker = tracker

        tracker.update(x)
        tracker.update_normalizer(normalizer)
        return x, normalizer


    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_output, norm_output):

        ctx.tracker.clear()
        return (grad_output, norm_output) + (None,)


class Block(nn.Module):

    def special_init_weights(self, config):
        self.down.weight.data.normal_(
            0.0, 1 / np.sqrt(config.residual_channels)
        )
        self.up.weight.data.normal_(
            0.0, 1 / np.sqrt(config.residual_channels)
        )


    def __init__(self, config, operation, tracker, num_inputs):
        super().__init__()

        self.hidden_size = config.hidden_size
        self.residual_channels = config.residual_channels
        self.num_inputs = num_inputs

        self.residual_size = self.hidden_size * self.residual_channels
        self.input_size = self.hidden_size * self.num_inputs

        self.operation = operation
        self.tracker = tracker

        self.normalizer_eps = config.normalizer_eps

        self.down = nn.Conv1d(
            self.residual_size, self.input_size,
            kernel_size=1, bias=False,
            groups=self.hidden_size
        )
        self.norm = nn.GroupNorm(
            self.num_inputs,
            self.input_size,
            eps=config.layer_norm_eps
        )
        self.up = nn.Conv1d(
            self.hidden_size, self.residual_size,
            kernel_size=1, bias=False,
            groups=self.hidden_size
        )

        self.debug = False


    def get_normalizer(self, ref):
        ones = torch.ones(1, self.hidden_size, 1, dtype=ref.dtype, device=ref.device)
        normalizer = self.up(ones)
        return normalizer

    
    def read(self, x, normalizer):
        bs, l, _ = x.shape

        x = x.view(bs*l, self.residual_size, 1)
        x = self.down(x)

        normalizer = normalizer.view(1, self.residual_size, 1)
        normalizer = self.down(normalizer)
        x = x / (normalizer + self.normalizer_eps)

        x = (
            x
            .view(bs, l, self.hidden_size, self.num_inputs)
            .permute(0, 1, 3, 2)
            .reshape(bs, l, self.input_size)
        )

        x = x.view(bs*l, self.input_size)
        x = self.norm(x)
        x = x.view(bs, l, self.input_size)

        return x
    

    def write(self, x):
        bs, l, _ = x.shape

        x = x.view(bs*l, self.hidden_size, 1)
        x = self.up(x)

        x = x.view(bs, l, self.residual_size)

        return x


    def compute(self, x, kwargs):
        return self.operation(x, **kwargs)


    def forward(self, x, normalizer, kwargs):
        if self.debug:
            return self.debug_forward(x, normalizer, kwargs)
        return bootstrap_fn.apply(x, normalizer, self, self.tracker, kwargs) # torch.is_grad_enabled())


    def debug_forward(self, s, normalizer, kwargs):
        
        x = self.read(s, normalizer)
        y = self.compute(x, kwargs)
        a = self.write(y)

        s_new = s + a
        new_norm = normalizer + self.get_normalizer(s_new)

        return s_new, new_norm


class bootstrap_fn(torch.autograd.Function):    

    @staticmethod
    def forward(ctx, s, normalizer, block: Block, tracker, kwargs):

        ctx.gpu_autocast_kwargs = {
            "enabled": torch.is_autocast_enabled(),
            "dtype": torch.get_autocast_gpu_dtype(),
            "cache_enabled": torch.is_autocast_cache_enabled()
        }
        ctx.cpu_autocast_kwargs = {
            "enabled": torch.is_autocast_cpu_enabled(),
            "dtype": torch.get_autocast_cpu_dtype(),
            "cache_enabled": torch.is_autocast_cache_enabled()
        }

        # get x (no grad, s not saved)
        x = block.read(s, normalizer)
        x.requires_grad = True

        # get y, and link to x
        with torch.enable_grad():
            y = block.compute(x, kwargs)

        # get a (no grad, doesn't matter we save y anyway)
        a = block.write(y)

        # add residual
        s_new = s + a

        # add keys
        new_norm = normalizer + block.get_normalizer(s)

        # save things
        ctx.save_for_backward(x, y)
        ctx.block = block
        ctx.tracker = tracker
        tracker.update(s_new)
        tracker.update_normalizer(new_norm)

        s_new.requires_grad = True
        new_norm.requires_grad = True
        return s_new, new_norm

    
    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_output, norm_output):
        x, y = ctx.saved_tensors

        if constants.XLA_AVAILABLE:
            xm.optimization_barrier_(
                _extract_tensors_from_list(
                    [x, y, grad_output, norm_output] +
                    list(ctx.block.parameters()) +
                    list(ctx.block.buffers()) +
                    [ctx.tracker.get(), ctx.tracker.get_normalizer()]
                )
            )

        # get a, populate x with gradient
        with torch.enable_grad(), \
            torch.cuda.amp.autocast(**ctx.gpu_autocast_kwargs), \
            torch.cpu.amp.autocast(**ctx.cpu_autocast_kwargs):
                a = ctx.block.write(y)
                new_norm = ctx.block.get_normalizer(a)
        torch.autograd.backward(a, grad_output)
        torch.autograd.backward(new_norm, norm_output)

        # reconstruct orignal input
        s = (ctx.tracker.get() - a).detach()
        normalizer = (ctx.tracker.get_normalizer() - new_norm).detach()

        # reconstruct x and populate s with gradient
        with torch.enable_grad(), \
            torch.cuda.amp.autocast(**ctx.gpu_autocast_kwargs), \
            torch.cpu.amp.autocast(**ctx.cpu_autocast_kwargs):
                s.requires_grad = True
                normalizer.requires_grad = True
                new_x = ctx.block.read(s, normalizer)
        torch.autograd.backward(new_x, x.grad)

        # save s for previous block
        ctx.tracker.update(s)
        ctx.tracker.update_normalizer(normalizer)

        # add grad to residual
        grad_output = grad_output + s.grad
        norm_output = norm_output + normalizer.grad

        return (grad_output, norm_output) + (None,)*3


class RatEmbedding(nn.Module):

    def special_init_weights(self, config):
        self.up.weight.data.normal_(
            0.0, 1 / np.sqrt(config.residual_channels)
        )


    def get_normalizer(self, ref):
        ones = torch.ones(1, self.hidden_size, 1, dtype=ref.dtype, device=ref.device)
        normalizer = self.up(ones)
        return normalizer


    def __init__(self, num_embeddings, hidden_size, residual_channels, tracker):
        super().__init__()

        self.num_embeddings = num_embeddings
        self.hidden_size = hidden_size

        self.residual_channels = residual_channels
        self.residual_size = hidden_size * residual_channels

        self.tracker = tracker

        self.embedding = nn.Embedding(
            num_embeddings, hidden_size
        )

        self.up = nn.Conv1d(
            self.hidden_size, self.residual_size,
            kernel_size=1, bias=False,
            groups=self.hidden_size
        )
    

    def forward(self, input_ids):
        bs, l = input_ids.shape

        out = self.embedding(input_ids)

        out = out.view(bs*l, self.hidden_size, 1)
        out = self.up(out)

        out = out.view(bs, l, self.residual_size)

        return out


class RatAttention(BaseAttention):

    def init_qkv_proj(self, config):
        self.qkv_proj = nn.Conv1d(
            3 * config.hidden_size, 3 * config.hidden_size,
            kernel_size=1, bias=config.use_qkv_bias,
            groups=3
        )
    

    def get_qkv(self, hidden_states):
        bs, l, _ = hidden_states.shape

        hidden_states = hidden_states.view(bs*l, 3 * self.hidden_size, 1)
        hidden_states = self.qkv_proj(hidden_states)
        hidden_states = hidden_states.view(bs, l, 3 * self.hidden_size)

        return hidden_states.chunk(3, dim=-1)


class RatMLP(BaseMLP):

    def init_mlp_input(self, config):
        self.in_proj = nn.Conv1d(
            2*config.hidden_size, 2*config.mlp_size,
            kernel_size=1, bias=False,
            groups=2
        )

    
    def get_mlp_input(self, hidden_states):
        bs, l, _ = hidden_states.shape

        hidden_states = hidden_states.view(bs*l, 2 * self.hidden_size, 1)
        hidden_states = self.in_proj(hidden_states)
        hidden_states = hidden_states.view(bs, l, 2 * self.mlp_size)

        return hidden_states.chunk(2, dim=-1)


class RatLayer(nn.Module):


    def special_init_weights(self, config: BaseConfig):
        if config.identity_init:
            raise ValueError("identity_init not supported for RatLayer!")

        self.attn_block.special_init_weights(config)
        self.mlp_block.special_init_weights(config)


    def post_step(self):
        pass


    def enable_debug(self):
        self.attn_block.debug = True
        self.mlp_block.debug = True


    def __init__(self, config: BaseConfig, layer_idx: int, tracker):
        super().__init__()

        self.hidden_size = config.hidden_size

        attn = RatAttention(config, layer_idx)
        mlp = RatMLP(config)

        self.attn_block = Block(config, attn, tracker, 3)
        self.mlp_block = Block(config, mlp, tracker, 2)


    def forward(
        self,
        hidden_states: torch.Tensor,
        normalizer: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value=None,
    ):

        # Self Attention
        hidden_states, normalizer = self.attn_block(
            hidden_states,
            normalizer,
            {
                'position_ids': position_ids,
                'attention_mask': attention_mask,
                'past_key_value': past_key_value
            }
        )

        # GLU MLP
        hidden_states, normalizer = self.mlp_block(
            hidden_states, 
            normalizer,
            {}
        )

        return hidden_states, normalizer


class RatTransformer(BaseTransformer):
    
    layer_type = RatLayer


    def get_extras(self, config):
        self.final_down = nn.Conv1d(
            self.residual_size, config.hidden_size,
            kernel_size=1, bias=False,
            groups=config.hidden_size
        )
        self.normalizer_eps = config.normalizer_eps

        self.norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)


    def special_init_weights(self, config):
        super().special_init_weights(config)

        self.vocab_embs.special_init_weights(config)

        self.final_down.weight.data.normal_(
            0.0, 1 / np.sqrt(config.residual_channels)
        )


    def enable_debug(self):
        for layer in self.layers:
            layer.enable_debug()


    def __init__(self, config: RatConfig):
        nn.Module.__init__(self)

        # info
        self.vocab_size = config.vocab_size
        self.max_sequence_length = config.max_sequence_length
        
        self.hidden_size = config.hidden_size
        self.residual_channels = config.residual_channels
        self.residual_size = self.hidden_size * self.residual_channels

        self.tracker = Tracker()

        # weights
        self.vocab_embs = RatEmbedding(
            config.vocab_size, config.hidden_size,
            config.residual_channels, self.tracker
        )
        
        self.use_rope = config.use_rope
        if not config.use_rope:
            raise ValueError("ONLY Rope supported for RatTransformer!")
        
        self.layers = nn.ModuleList(
            [
                RatLayer(config, i, self.tracker)
                for i in range(config.num_layers)
            ]
        )
        
        self.get_extras(config)

        # Compute configuration
        self.gradient_checkpointing = config.gradient_checkpointing
        self.gradient_checkpointing_layers = config.gradient_checkpointing_layers
        if self.gradient_checkpointing:
            raise ValueError("Gradient Checkpointing not supported for RatTransformer!")

        self.debug = config.bootstrap_debug
        if self.debug:
            self.enable_debug()


    def get_output(
        self,
        hidden_states: torch.Tensor,
        normalizer: torch.Tensor
    ):
        bs, l, _ = hidden_states.shape

        hidden_states = hidden_states.view(bs*l, self.residual_size, 1)
        hidden_states = self.final_down(hidden_states)

        normalizer = normalizer.view(1, self.residual_size, 1)
        normalizer = self.final_down(normalizer)
        hidden_states = hidden_states / (normalizer + self.normalizer_eps)

        hidden_states = hidden_states.view(bs, l, self.hidden_size)
        hidden_states = self.norm(hidden_states)

        return hidden_states
    

    def forward(
        self,
        input_ids: torch.Tensor,
        segment_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        kv=None
    ):

        # get inputs
        position_ids = self._get_position_ids(input_ids, position_ids)
        attention_mask = self._get_mask(input_ids, segment_ids)
        hidden_states = self.get_hidden_states(input_ids, position_ids)

        # start tracking
        self.tracker.clear()
        normalizer = self.vocab_embs.get_normalizer(hidden_states)
        hidden_states, normalizer = self.tracker(hidden_states, normalizer)

        for layer in self.layers:
            hidden_states, normalizer = layer(
                hidden_states=hidden_states,
                normalizer=normalizer,
                position_ids=position_ids,
                attention_mask=attention_mask,
                past_key_value=kv
            )

        hidden_states = self.get_output(hidden_states, normalizer)

        return hidden_states


class RatLmModel(BaseLmModel):

    transformer_type = RatTransformer
    