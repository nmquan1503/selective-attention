import torch
import torch.nn.functional as F
from typing import Tuple

def check_ndim(
    tensor: torch.Tensor | None,
    ndim: int,
    name: str | None = None,
    optional: bool = False
):
    if tensor is None:
        if not optional:
            message = "" if name else f"[{name}] "
            message += f"Expected tensor with ndim {ndim}, but got None"
            raise ValueError(message)
        return

    if tensor.ndim != ndim:
        prefix = f"[{name}] " if name else ""
        raise ValueError(
            f"{prefix}Dim mismatch: expected ndim={ndim}, got ndim={tensor.ndim}, shape={tuple(tensor.shape)}"
        )

def check_shape(
    tensor: torch.Tensor | None, 
    shape: Tuple[int], 
    name: str | None = None, 
    optional: bool = False
):
    if tensor is None:
        if not optional:
            message = "" if name else  f"[{name}] "
            message += f"Expected tensor with shape {shape}, but got None"
            raise ValueError(message)
    elif tensor.shape != shape:
        message = "" if name else  f"[{name}] "
        message += f"Shape mismatch: expected {shape}, got {tuple(tensor.shape)}"
        raise ValueError(message)

def to_contiguous(tensor: torch.Tensor | None):
    if tensor is None:
        return tensor
    if not tensor.is_contiguous():
        return tensor.contiguous()
    return tensor


def right_align(
    x: torch.Tensor,
    valid_mask: torch.Tensor,
    buffer_size: int = 0,
    return_new_mask: bool = False
):
    """
    Args:
        x:          (batch_size, num_heads, seq_len, head_dim)
        valid_mask: (batch_size, num_heads, seq_len)

    Returns:
        x:          (batch_size, num_heads, max_valid + buffer_size, head_dim)
        valid_mask: (batch_size, num_heads, max_valid + buffer_size)
    """
    batch_size, num_heads, seq_len, head_dim = x.shape
    device = x.device

    num_valid = valid_mask.sum(dim=-1)
    max_valid = num_valid.max().item()
    rank = valid_mask.cumsum(dim=-1) - 1
    shift = max_valid - num_valid
    new_len = max_valid + buffer_size

    out = torch.zeros(batch_size, num_heads, new_len, head_dim, device=device, dtype=x.dtype)

    flat_bh = batch_size * num_heads
    out_flat = out.view(flat_bh, new_len, head_dim)
    x_flat = x.view(flat_bh, seq_len, head_dim)
    valid_flat = valid_mask.view(flat_bh, seq_len)
    dst_idx = rank + shift.unsqueeze(-1)
    dst_flat = dst_idx.view(flat_bh, seq_len)

    sel_idx = dst_flat[valid_flat]
    bh_idx = torch.arange(flat_bh, device=device).unsqueeze(-1).expand(-1, seq_len)
    bh_idx_sel = bh_idx[valid_flat]

    out_flat[bh_idx_sel, sel_idx] = x_flat[valid_flat]

    if not return_new_mask:
        return out

    shift_flat = shift.view(flat_bh, 1)
    num_valid_flat = num_valid.view(flat_bh, 1)
    pos_idx = torch.arange(new_len, device=device)

    new_valid_flat = (pos_idx >= shift_flat) & (pos_idx < shift_flat + num_valid_flat)
    new_valid = new_valid_flat.view(batch_size, num_heads, new_len)

    return out, new_valid


def compress(
    x: torch.Tensor,
    valid_mask: torch.Tensor,
    buffer_size: int = 0,
    return_new_mask: bool = False
):
    """    
    Args:
        x: (batch_size, num_heads, seq_len, head_dim)
        valid_mask: (batch_size, num_heads, seq_len)
    
    Returns:
        out: (batch_size, num_heads, max_valid + buffer_size, head_dim)
        new_valid:  (batch_size, num_heads, max_valid + buffer_size)
    """
    batch_size, num_heads, seq_len, head_dim = x.shape
    device = x.device

    num_valid = valid_mask.sum(dim=-1)
    max_valid = num_valid.max().item()

    rank = valid_mask.cumsum(dim=-1) - 1
    new_len = max_valid + buffer_size

    out = torch.zeros(batch_size, num_heads, new_len, head_dim, device=device, dtype=x.dtype)

    flat_bh = batch_size * num_heads
    out_flat = out.view(flat_bh, new_len, head_dim)
    x_flat = x.view(flat_bh, seq_len, head_dim)
    valid_flat = valid_mask.view(flat_bh, seq_len)
    rank_flat = rank.view(flat_bh, seq_len)

    sel_idx = rank_flat[valid_flat]
    bh_idx = torch.arange(flat_bh, device=device).unsqueeze(-1).expand(-1, seq_len)
    bh_idx_sel = bh_idx[valid_flat]

    out_flat[bh_idx_sel, sel_idx] = x_flat[valid_flat]

    if not return_new_mask:
        return out

    num_valid_flat = num_valid.view(flat_bh, 1)
    pos_idx = torch.arange(new_len, device=device).view(1, -1)
    new_valid_flat = pos_idx < num_valid_flat
    new_valid = new_valid_flat.view(batch_size, num_heads, new_len)

    return out, new_valid

def pad_buffer(x: torch.Tensor, buffer_size: int):
    """
    Args:
        x: (batch_size, num_heads, seq_len, head_dim)
    
    Returns:
        x: (batch_size, num_heads, seq_len + buffer_size, head_dim)
    """
    if buffer_size <= 0:
        return x
    return F.pad(
        x, 
        pad=(
            0, 0,
            0, buffer_size,
            0, 0
        ),
        mode="constant",
        value=0
    )