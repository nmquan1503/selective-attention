import torch
import math
import triton
from typing import Tuple

from .forward_kernel import (
    chunk_cumsum_forward_kernel,
    chunk_state_forward_kernel,
    state_passing_forward_kernel,
    bmm_chunk_forward_kernel,
    chunk_scan_forward_kernel
)

def chunk_cumsum_forward(
    A: torch.Tensor,
    delta_raw: torch.Tensor,
    delta_bias: torch.Tensor | None,
    length: torch.Tensor,
    chunk_size: int,
    use_delta_softplus: bool,
    delta_limit: Tuple = (0.0, float("inf"))
):
    """
    Args:
        A: (num_heads)
        delta_raw: (batch_size, seq_len, num_heads)
        delta_bias: (num_heads,)
        length: (batch_size,)

    Returns:
        decay_cumsum: (batch_size, num_heads, num_chunks, chunk_size)
        delta: (batch_size, num_heads, num_chunks, chunk_size)
    """
    batch_size, seq_len, num_heads = delta_raw.shape
    device = delta_raw.device
    num_chunks = math.ceil(seq_len / chunk_size)
    
    delta = torch.empty(batch_size, num_heads, num_chunks, chunk_size, device=device, dtype=torch.float32)
    decay_cumsum = torch.empty(batch_size, num_heads, num_chunks, chunk_size, device=device, dtype=torch.float32)
    
    grid = lambda META: (batch_size, num_chunks, triton.cdiv(num_heads, META["HEAD_GROUP_SIZE"]))
    chunk_cumsum_forward_kernel[grid](
        A, delta_raw, delta_bias, delta, decay_cumsum, length,
        seq_len, chunk_size, num_heads, delta_limit[0], delta_limit[1],
        *(A.stride()),
        *(delta_raw.stride()),
        *(delta_bias.stride() if delta_bias is not None else (0,)),
        *(delta.stride()),
        *(decay_cumsum.stride()),
        *(length.stride()),
        use_delta_softplus,
        HAS_DELTA_BIAS=delta_bias is not None,
        CHUNK_SIZE_ALIGNED=triton.next_power_of_2(chunk_size)
    )

    return decay_cumsum, delta


####################################################################################################


def chunk_state_forward(
    u: torch.Tensor,
    B: torch.Tensor,
    delta: torch.Tensor,
    decay_cumsum: torch.Tensor,
):
    """
    Args:
        u: (batch_size, seq_len, num_heads, head_dim)
        B: (batch_size, seq_len, num_groups, state_dim)
        delta: (batch_size, num_heads, num_chunks, chunk_size)
        decay_cumsum: (batch_size, num_heads, num_chunks, chunk_size)
    
    Returns:
        h: (batch_size, num_chunks, num_heads, head_dim, state_dim)
    """
    batch_size, seq_len, num_heads, head_dim = u.shape
    _, _, num_groups, state_dim = B.shape
    _, _, num_chunks, chunk_size = delta.shape
    device = u.device

    h = torch.empty(batch_size, num_chunks, num_heads, head_dim, state_dim, device=device, dtype=torch.float32)

    grid = lambda META: (triton.cdiv(head_dim, META["HEAD_TILE_SIZE"]) * triton.cdiv(state_dim, META["STATE_TILE_SIZE"]), batch_size * num_chunks, num_heads)
    chunk_state_forward_kernel[grid](
        u, B, h, delta, decay_cumsum,
        batch_size, seq_len, chunk_size, head_dim, state_dim, num_heads // num_groups,
        *(u.stride()),
        *(B.stride()),
        *(h.stride()),
        *(delta.stride()),
        *(decay_cumsum.stride())
    )

    return h


####################################################################################################


def state_passing_forward(
    h: torch.Tensor,
    h_init: torch.Tensor | None,
    decay_last: torch.Tensor,
):
    """
    Args:
        h: (batch_size, num_chunks, num_heads, head_dim * state_dim)
        h_init: (batch_size, num_heads, head_dim * state_dim)
        decay_last: (batch_size, num_heads, num_chunks)
    
    Returns:
        h: (batch_size, num_chunks, num_heads, head_dim * state_dim)
        h_last: (batch_size, num_heads, head_dim * state_dim)
    """
    batch_size, num_chunks, num_heads, head_state_dim = h.shape
    device = h.device

    h_last = torch.empty((batch_size, num_heads, head_state_dim), device=device, dtype=torch.float32)

    grid = lambda META: (batch_size, num_heads, triton.cdiv(head_state_dim, META["HEAD_STATE_GROUP_SIZE"]))
    state_passing_forward_kernel[grid](
        h, h_init, h_last, decay_last,
        num_chunks, head_state_dim,
        *(h.stride()),
        *(h_init.stride() if h_init is not None else (0, 0, 0)),
        *(h_last.stride()),
        *(decay_last.stride()),
        HAS_H_INIT=h_init is not None
    )

    return h, h_last


####################################################################################################


def bmm_chunk_forward(
    B: torch.Tensor,
    C: torch.Tensor,
    chunk_size: int,
    is_causal: bool = False
):
    """
    Args:
        B, C: (batch_size, seq_len, num_groups, state_dim)
    
    Returns:
        CB: (batch_size, num_chunks, num_groups, chunk_size, chunk_size)
    """

    batch_size, seq_len, num_groups, state_dim = B.shape
    num_chunks = math.ceil(seq_len / chunk_size)
    device = B.device

    CB = torch.empty((batch_size, num_chunks, num_groups, chunk_size, chunk_size), device=device, dtype=torch.float32)

    grid = lambda META: (triton.cdiv(chunk_size, META["CHUNK_TILE_Y_SIZE"]) * triton.cdiv(chunk_size, META["CHUNK_TILE_X_SIZE"]), batch_size, num_chunks * num_groups)
    bmm_chunk_forward_kernel[grid](
        B, C, CB,
        seq_len, chunk_size, num_groups, state_dim,
        *(B.stride()),
        *(C.stride()),
        *(CB.stride()),
        is_causal,
    )

    return CB


####################################################################################################


def chunk_scan_forward(
    u: torch.Tensor,
    delta: torch.Tensor,
    decay_cumsum: torch.Tensor,
    C: torch.Tensor,
    h: torch.Tensor,
    CB: torch.Tensor,
):
    """
    Args:
        u: (batch_size, seq_len, num_heads, head_dim)
        delta: (batch_size, num_heads, num_chunks, chunk_size)
        decay_cumsum: (batch_size, num_heads, num_chunks, chunk_size)
        C: (batch_size, seq_len, num_groups, state_dim)
        h: (batch_size, num_chunks, num_heads, head_dim, state_dim)
        CB: (batch_size, num_chunks, num_groups, chunk_size, chunk_size)
    
    Returns:
        y: (batch_size, seq_len, num_heads, head_dim)
    """
    batch_size, seq_len, num_heads, head_dim = u.shape
    _, _, num_chunks, chunk_size = delta.shape
    _, _, num_groups, state_dim = C.shape
    device = u.device

    y = torch.empty(batch_size, seq_len, num_heads, head_dim, device=device, dtype=torch.float32)

    grid = lambda META: (triton.cdiv(chunk_size, META["CHUNK_TILE_SIZE"]) * triton.cdiv(head_dim, META["HEAD_TILE_SIZE"]), batch_size * num_chunks, num_heads)
    chunk_scan_forward_kernel[grid](
        u, delta, decay_cumsum, C, h, CB, y,
        batch_size, seq_len, chunk_size, head_dim, state_dim, num_heads // num_groups,
        *(u.stride()),
        *(delta.stride()),
        *(decay_cumsum.stride()),
        *(C.stride()),
        *(h.stride()),
        *(CB.stride()),
        *(y.stride()),
        IS_CAUSAL=True,
        STATE_DIM_ALIGNED=max(triton.next_power_of_2(state_dim), 16)
    )

    return y


####################################################################################################


def ssd_scan_forward(
    u: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    delta_raw: torch.Tensor,
    delta_bias: torch.Tensor | None,
    h_init: torch.Tensor | None,
    length: torch.Tensor,
    chunk_size: int,
    use_delta_softplus: bool = True,
    delta_limit: Tuple = (0.0, float("inf")),
    is_causal: bool = True
):
    """
    Args:
        u: (batch_size, seq_len, num_heads, head_dim)
        A: (num_heads,)
        B, C: (batch_size, seq_len, num_groups, state_dim)
        delta_raw: (batch_size, seq_len, num_heads)
        delta_bias: (num_heads,)
        h_init: (batch_size, num_heads, head_dim, state_dim)
        length: (batch_size,)
    
    Returns:
        y: (batch_size, seq_len, num_heads, head_dim)
        h_last: (batch_size, num_heads, head_dim, state_dim)
    """
    batch_size, seq_len, num_heads, head_dim = u.shape
    _, _, num_groups, state_dim = B.shape
    num_chunks = math.ceil(seq_len / chunk_size)

    decay_cumsum, delta = chunk_cumsum_forward(
        A, delta_raw, delta_bias, length,
        chunk_size, use_delta_softplus, delta_limit
    )
    
    h = chunk_state_forward(u, B, delta, decay_cumsum)
    
    decay_last = decay_cumsum[:, :, :, -1]
    h, h_last = state_passing_forward(
        h.view(batch_size, num_chunks, num_heads, -1), 
        h_init.view(batch_size, num_heads, -1) if h_init is not None else None, 
        decay_last
    )

    h = h.view(batch_size, num_chunks, num_heads, head_dim, -1)
    h_last = h_last.view(batch_size, num_heads, head_dim, -1)
    
    CB = bmm_chunk_forward(B, C, chunk_size, is_causal)
    
    y = chunk_scan_forward(u, delta, decay_cumsum, C, h, CB)

    return y, h_last