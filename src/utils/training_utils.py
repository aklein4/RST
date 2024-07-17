from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def loss(
    logits: torch.Tensor,
    x: torch.LongTensor,
    ignore_index: Optional[int]=-1,
    eos_index: Optional[int]=None
) -> torch.Tensor:
    """ Standard cross-entropy loss for language modeling.
     - applies offset so that logits_{t} predicts x_{t+1}
     - ignores padding tokens and last logits
     
    Args:
        logits (torch.Tensor): token logits from model [B, T, V]
        x (torch.LongTensor): target tokens [B, T]
        ignore_index (Optional[int], optional): Paddding token to ignore. Defaults to -1.

    Returns:
        torch.Tensor: cross-entropy loss [nats]
    """
    x_inp = x[:, :-1].view(-1)
    x = x[:, 1:].view(-1)
    logits = logits[:, :-1].view(-1, logits.shape[-1])
    
    mask = x != ignore_index
    if eos_index is not None:
        mask = mask & (x_inp != eos_index)

    # we assume logits are normalized, to save (quite a bit of) memory
    ar = torch.arange(x.shape[0], device=x.device, dtype=x.dtype)
    loss = -logits[ar, x]

    # mask padding tokens
    loss = torch.masked_fill(loss, ~mask, 0.0)
    return loss.sum() / mask.float().sum()


@torch.no_grad()
def ppl(
    logits: torch.Tensor,
    x: torch.LongTensor,
    ignore_index: Optional[int]=-1,
    eos_index: Optional[int]=None
) -> torch.Tensor:
    """ Compute perplexity of the model.
     - uses same data logic as loss()

    Args:
        logits (torch.Tensor): token logits from model [B, T, V]
        x (torch.LongTensor): target tokens [B, T]
        ignore_index (Optional[int], optional): Paddding token to ignore. Defaults to -1.

    Returns:
        torch.Tensor: Perplexity [nats]
    """
    x_inp = x[:, :-1]
    x = x[:, 1:]
    logits = logits[:, :-1]

    mask = x != ignore_index
    if eos_index is not None:
        mask = mask & (x_inp != eos_index)

    logp = -F.cross_entropy(
        logits.contiguous().view(-1, logits.shape[-1]),
        x.contiguous().view(-1),
        reduction='none'
    ).view(x.shape)

    logp = torch.masked_fill(logp, ~mask, 0.0)
    logp_seq = logp.sum(-1) / (mask).float().sum(-1)

    return torch.exp(-logp_seq).mean()


@torch.no_grad()
def acc(
    logits: torch.Tensor,
    x: torch.LongTensor,
    ignore_index: Optional[int]=-1,
    eos_index: Optional[int]=None
) -> torch.Tensor:
    """ Compute top-1 next-token accuracy of the model.
     - uses same data logic as loss()
    
    Args:
        logits (torch.Tensor): logits from model [B, T, V]
        x (torch.LongTensor): target tokens [B, T]
        ignore_index (Optional[int], optional): Paddding token to ignore. Defaults to -1.

    Returns:
        torch.Tensor: top-1 token accuracy
    """
    x_inp = x[:, :-1]
    x, logits = x[:, 1:], logits[:, :-1]

    mask = x != ignore_index
    if eos_index is not None:
        mask = mask & (x_inp != eos_index)

    corr = torch.logical_and(
        logits.argmax(-1) == x,
        mask
    ).float().sum()
    return corr / (mask).float().sum()


@torch.no_grad()
def pcorr(
    logits: torch.Tensor,
    x: torch.LongTensor,
    ignore_index: Optional[int]=-1,
    eos_index: Optional[int]=None
) -> torch.Tensor:
    """ Compute token prediction probability of the model.
     - measures probability that a token sampled from logits is equal to target token
     - uses same data logic as loss()

    Args:
        logits (torch.Tensor): logits from model [B, T, V]
        x (torch.LongTensor): target tokens [B, T]
        ignore_index (Optional[int], optional): Paddding token to ignore. Defaults to -1.

    Returns:
        torch.Tensor: next-token prediction probability
    """
    x_inp = x[:, :-1].view(-1)
    x = x[:, 1:].contiguous().view(-1)
    logits = logits[:, :-1].contiguous().view(-1, logits.shape[-1])
    
    mask = x != ignore_index
    if eos_index is not None:
        mask = mask & (x_inp != eos_index)

    logp = -F.cross_entropy(
        logits, x,
        reduction='none'
    )
    p = torch.exp(logp)

    p = torch.masked_fill(p, ~mask, 0.0)
    return p.sum() / (mask).float().sum()
