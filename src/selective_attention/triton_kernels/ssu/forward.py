import torch
from typing import Tuple
import triton

from .forward_kernel import ssu_forward_kernel

def ssu_forward(
    u: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    delta_raw: torch.Tensor,
    delta_bias: torch.Tensor,
    h: torch.Tensor,
    use_delta_softplus: bool = True,
    delta_limit: Tuple = (0.0, float("inf"))
):
    """
    Args:
        u: (batch_size, num_heads, head_dim)
        A: (num_heads,)
        B, C: (batch_size, num_groups, state_dim)
        delta_raw: (batch_size, num_heads)
        delta_bias: (num_heads,)
        h: (batch_size, num_heads, head_dim, state_dim)

    Returns:
        y: (batch_size, num_heads, head_dim)
        h: (batch_size, num_heads, head_dim, state_dim)
    """
    batch_size, num_heads, head_dim = u.shape
    _, num_groups, state_dim = B.shape

    y = torch.empty_like(u)
    STATE_DIM_ALIGNED = triton.next_power_of_2(state_dim)
    HEAD_TILE_SIZE, num_warps = (
        (32, 4) if state_dim <= 16 else
        (16, 4) if state_dim <= 32 else
        (8, 4) if state_dim <= 64 else
        (4, 4) if state_dim <= 128 else
        (4, 8)
    )

    grid = lambda META: (triton.cdiv(head_dim, HEAD_TILE_SIZE), batch_size, num_heads)
    ssu_forward_kernel[grid](
        u, A, B, C, delta_raw, delta_bias, h, y,
        batch_size, num_heads, head_dim, state_dim, num_heads // num_groups, delta_limit[0], delta_limit[1],
        *(u.stride()),
        *(A.stride()),
        *(B.stride()),
        *(C.stride()),
        *(delta_raw.stride()),
        *(delta_bias.stride() if delta_bias is not None  else (0,)),
        *(h.stride()),
        *(y.stride()),
        HAS_DELTA_BIAS=delta_bias is not None,
        USE_DELTA_SOFTPLUS=use_delta_softplus,
        STATE_DIM_ALIGNED=STATE_DIM_ALIGNED,
        HEAD_TILE_SIZE=HEAD_TILE_SIZE,
        num_warps=num_warps
    )
    
    return y, h