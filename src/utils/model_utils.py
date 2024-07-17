import torch
import torch.nn as nn
import torch.nn.functional as F


class RotaryEmbedding(nn.Module):

    def __init__(self, total_dim, frac, max_position_embeddings, base):
        super().__init__()

        assert total_dim % frac == 0, f'dimension {total_dim} must be divisible by frac {frac}'
        self.total_dim = total_dim
        self.dim = total_dim // frac
        assert self.dim % 2 == 0, f'dimension {self.dim} must be divisible by 2'

        self.max_position_embeddings = max_position_embeddings
        self.base = base

        # inverse frequencies for rotations
        freq_ar = torch.arange(0, self.dim, 2).float()
        inv_freq = (
            1.0 /
            (self.base ** (freq_ar / self.dim))
        ) # [D/2]

        # only use integer positions, so we cache sin/cos as embeddings
        pos = torch.arange(0, self.max_position_embeddings).float()
        freqs = torch.matmul(inv_freq[:, None], pos[None, :]) # [D/2, L]
        freqs = freqs.permute(1, 0) # [L, D/2]

        freqs = torch.cat((freqs, freqs), dim=-1) # [L, D]
        sin = freqs.sin()
        cos = freqs.cos()
        
        self.sin_emb = nn.Embedding(self.max_position_embeddings, self.dim)
        self.sin_emb.weight.data = sin.contiguous()

        self.cos_emb = nn.Embedding(self.max_position_embeddings, self.dim)
        self.cos_emb.weight.data = cos.contiguous()


    def _get_sin_cos(self, position_ids):
        return (
            self.sin_emb(position_ids).detach(),
            self.cos_emb(position_ids).detach()
        )


    def _rotate_half(self, x):
        """Rotates half the hidden dims of the input."""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)


    def forward(self, q, k, position_ids):
        assert q.shape[-1] == self.total_dim, f'q shape {q.shape} does not match total_dim {self.total_dim}'
        assert k.shape[-1] == self.total_dim, f'k shape {k.shape} does not match total_dim {self.total_dim}'

        sin, cos = self._get_sin_cos(position_ids)
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)

        if self.dim == self.total_dim:
            q = (q * cos) + (self._rotate_half(q) * sin)
            k = (k * cos) + (self._rotate_half(k) * sin)
            return q, k

        q_rot, q_no_rot = q[..., : self.dim], q[..., self.dim :]
        k_rot, k_no_rot = k[..., : self.dim], k[..., self.dim :]

        q_rot = (q_rot * cos) + (self._rotate_half(q_rot) * sin)
        k_rot = (k_rot * cos) + (self._rotate_half(k_rot) * sin)

        q = torch.cat((q_rot, q_no_rot), dim=-1)
        k = torch.cat((k_rot, k_no_rot), dim=-1)

        return q, k
    