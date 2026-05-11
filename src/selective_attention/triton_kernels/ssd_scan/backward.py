import torch
import math
import triton
from typing import Tuple

from .forward import (
    chunk_cumsum_forward,
    chunk_state_forward,
    state_passing_forward,
    bmm_chunk_forward,
)
from .backward_kernel import (
    chunk_scan_backward_h_kernel,
    state_passing_backward_kernel,
    chunk_scan_chunk_state_backward_u_kernel,
    chunk_state_backward_B_kernel,
    chunk_scan_backward_C_grad_kernel,
    chunk_scan_backward_CB_kernel,
    bmm_chunk_backward_kernel,
    chunk_scan_backward_decay_cumsum_kernel,
    chunk_cumsum_backward_kernel
)


def chunk_scan_backward_h(
    C: torch.Tensor,
    decay_cumsum: torch.Tensor,
    y_grad: torch.Tensor,
    length: torch.Tensor
) -> torch.Tensor:
    """
    Args:
        C: (batch_size, seq_len, num_groups, state_dim)
        decay_cumsum: (batch_size, num_heads, num_chunks, chunk_size)
        y_grad: (batch_size, seq_len, num_heads, head_dim)
        length: (batch_size,)
    
    Returns:
        h_grad: (batch_size, num_chunks, num_heads, head_dim, state_dim)
    """
    batch_size, seq_len, num_heads, head_dim = y_grad.shape
    _, _, num_chunks, chunk_size = decay_cumsum.shape
    _, _, num_groups, state_dim = C.shape
    device = C.device

    h_grad = torch.empty(batch_size, num_chunks, num_heads, head_dim, state_dim, device=device, dtype=torch.float32)

    grid = lambda META: (triton.cdiv(head_dim, META["HEAD_TILE_SIZE"]) * triton.cdiv(state_dim, META["STATE_TILE_SIZE"]), batch_size * num_chunks, num_heads)
    chunk_scan_backward_h_kernel[grid](
        C, decay_cumsum, y_grad, h_grad, length,
        batch_size, seq_len, num_chunks, chunk_size, head_dim, state_dim, num_heads // num_groups,
        *(C.stride()),
        *(decay_cumsum.stride()),
        *(y_grad.stride()),
        *(h_grad.stride()),
        *(length.stride())
    )

    return h_grad


####################################################################################################

# ->
def state_passing_backward(
    h: torch.Tensor,
    decay_last: torch.Tensor,
    h_grad: torch.Tensor,
    h_last_grad: torch.Tensor,
    has_h_init: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """
    Args:
        h: (batch_size, num_chunks, num_heads, head_dim * state_dim)
        decay_last: (batch_size, num_heads, num_chunks)
        h_grad: (batch_size, num_chunks, num_heads, head_dim * state_dim)
        h_last_grad: (batch_size, num_heads, head_dim * state_dim)
    
    Returns:
        h_grad: (batch_size, num_chunks, num_heads, head_dim * state_dim)
        decay_last_grad: (batch_size, num_heads, num_chunks)
        h_init_grad: (batch_size, num_heads, head_dim * state_dim)
    """
    batch_size, num_chunks, num_heads, head_state_dim = h.shape
    device = h.device

    HEAD_STATE_GROUP_SIZE_MIN = 64
    num_head_state_groups = (head_state_dim + HEAD_STATE_GROUP_SIZE_MIN - 1) // HEAD_STATE_GROUP_SIZE_MIN
    decay_last_grad = torch.empty(batch_size, num_heads, num_chunks, num_head_state_groups, device=device, dtype=torch.float32)
    h_init_grad = torch.empty_like(h_grad[:, 0]) if has_h_init else None

    grid = lambda META: (triton.cdiv(head_state_dim, META["HEAD_STATE_GROUP_SIZE"]), batch_size, num_heads)
    state_passing_backward_kernel[grid](
        h, decay_last, h_grad, h_last_grad, decay_last_grad, h_init_grad,
        num_chunks, head_state_dim,
        *(h.stride()),
        *(decay_last.stride()),
        *(h_grad.stride()),
        *(h_last_grad.stride()),
        *(decay_last_grad.stride()),
        *(h_init_grad.stride() if has_h_init else (0, 0, 0)),
        HAS_H_INIT=has_h_init
    )
    HEAD_STATE_GROUP_SIZE_ACTUAL = state_passing_backward_kernel.best_config.kwargs["HEAD_STATE_GROUP_SIZE"]
    num_valid_head_state_groups = (head_state_dim + HEAD_STATE_GROUP_SIZE_ACTUAL - 1) // HEAD_STATE_GROUP_SIZE_ACTUAL
    decay_last_grad = decay_last_grad[..., :num_valid_head_state_groups].sum(dim=-1)
    
    return h_grad, decay_last_grad, h_init_grad


####################################################################################################


def chunk_scan_chunk_state_backward_u(
    u: torch.Tensor,
    delta: torch.Tensor,
    decay_cumsum: torch.Tensor,
    B: torch.Tensor,
    CB: torch.Tensor,
    y_grad: torch.Tensor,
    h_grad: torch.Tensor,
    length: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Args:
        u: (batch_size, seq_len, num_heads, head_dim)
        delta: (batch_size, num_heads, num_chunks, chunk_size)
        decay_cumsum: (batch_size, num_heads, num_chunks, chunk_size)
        B: (batch_size, seq_len, num_groups, state_dim)
        CB: (batch_size, num_chunks, num_groups, chunk_size, chunk_size)
        y_grad: (batch_size, seq_len, num_heads, head_dim)
        h_grad: (batch_size, num_chunks, num_heads, head_dim, state_dim)
        length: (batch_size,)

    Returns:
        u_grad: (batch_size, seq_len, num_heads, head_dim)
        delta_grad: (batch_size, num_heads, num_chunks, chunk_size)
    """
    batch_size, seq_len, num_heads, head_dim = u.shape
    _, _, num_chunks, chunk_size = delta.shape
    _, _, num_groups, state_dim = B.shape
    device = u.device

    u_grad = torch.empty_like(u)
    delta_grad = torch.empty_like(delta)

    grid = lambda META: (triton.cdiv(chunk_size, META["CHUNK_TILE_SIZE"]) * triton.cdiv(head_dim, META["HEAD_TILE_SIZE"]), batch_size * num_chunks, num_heads)
    chunk_scan_chunk_state_backward_u_kernel[grid](
        u, delta, decay_cumsum, B, CB, y_grad, h_grad, u_grad, delta_grad, length,
        batch_size, seq_len, chunk_size, head_dim, state_dim, num_heads // num_groups,
        *(u.stride()),
        *(delta.stride()),
        *(decay_cumsum.stride()),
        *(B.stride()),
        *(CB.stride()),
        *(y_grad.stride()),
        *(h_grad.stride()),
        *(u_grad.stride()),
        *(delta_grad.stride()),
        *(length.stride()),
        STATE_DIM_ALIGNED=max(triton.next_power_of_2(state_dim), 16)
    )
    
    return u_grad, delta_grad


####################################################################################################

# -> 
def chunk_state_backward_B(
    u: torch.Tensor,
    B: torch.Tensor,
    delta: torch.Tensor,
    decay_cumsum: torch.Tensor, 
    h_grad: torch.Tensor,
    length: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Args:
        u: (batch_size, seq_len, num_heads, head_dim)
        B: (batch_size, seq_len, num_groups, state_dim)
        delta: (batch_size, num_heads, num_chunks, chunk_size)
        decay_cumsum: (batch_size, num_heads, num_chunks, chunk_size)
        h_grad: (batch_size, num_chunks, num_heads, head_dim, state_dim)
        length: (batch_size,)
        
    Returns:
        B_grad: (batch_size, seq_len, num_groups, state_dim)
        decay_grad_from_B: (batch_size, num_heads, num_chunks, chunk_size)
    """
    batch_size, seq_len, num_heads, head_dim = u.shape
    _, _, num_groups, state_dim = B.shape
    _, _, num_chunks, chunk_size = delta.shape
    num_heads_per_group = num_heads // num_groups
    
    decay_grad_from_B = torch.empty_like(decay_cumsum)
    sm_count = torch.cuda.get_device_properties(u.device).multi_processor_count
    num_heads_per_program = max(min(math.ceil(batch_size * num_chunks * num_heads / sm_count), num_heads_per_group), 1)
    num_splits = triton.cdiv(num_heads_per_group, num_heads_per_program)
    B_grad = torch.empty(batch_size, seq_len, num_splits, num_groups, state_dim, device=u.device, dtype=torch.float32)

    grid = lambda META: (triton.cdiv(chunk_size, META["CHUNK_TILE_SIZE"]) * triton.cdiv(state_dim, META["STATE_TILE_SIZE"]), batch_size * num_chunks, num_splits * num_groups)
    chunk_state_backward_B_kernel[grid](
        u, B, delta, decay_cumsum, h_grad, decay_grad_from_B, B_grad, length,
        batch_size, seq_len, chunk_size, num_heads, head_dim, state_dim, num_groups, num_heads_per_program,
        *(u.stride()),
        *(B.stride()),
        *(delta.stride()),
        *(decay_cumsum.stride()),
        *(h_grad.stride()),
        *(decay_grad_from_B.stride()),
        *(B_grad.stride()),
        *(length.stride()),
        HEAD_DIM_ALIGNED=max(triton.next_power_of_2(head_dim), 16)
    )
    B_grad = B_grad.sum(2)
    torch.cumsum(decay_grad_from_B, dim=-1, out=decay_grad_from_B)
    
    return B_grad, decay_grad_from_B


####################################################################################################

# ->
def chunk_scan_backward_C(
    C: torch.Tensor,
    h: torch.Tensor,
    decay_cumsum: torch.Tensor,
    y_grad: torch.Tensor,
    length: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Args:
        C: (batch_size, seq_len, num_groups, state_dim)
        h: (batch_size, num_chunks, num_heads, head_dim, state_dim)
        decay_cumsum: (batch_size, num_heads, num_chunks, chunk_size)
        y_grad: (batch_size, seq_len, num_heads, head_dim)
        length: (batch_size,)
        
    Returns:
        C_grad: (batch_size, seq_len, num_groups, state_dim)
        decay_grad_from_C: (batch_size, num_heads, num_chunks, chunk_size)
    """
    batch_size, num_chunks, num_heads, head_dim, state_dim = h.shape
    _, seq_len, num_groups, _ = C.shape
    chunk_size = decay_cumsum.shape[-1]
    num_heads_per_group = num_heads // num_groups
    device = h.device

    decay_grad_from_C = torch.empty_like(decay_cumsum)
    sm_count = torch.cuda.get_device_properties(device).multi_processor_count
    num_heads_per_program = max(min(math.ceil(batch_size * num_chunks * num_heads / sm_count), num_heads_per_group), 1)
    num_splits = triton.cdiv(num_heads_per_group, num_heads_per_program)
    C_grad = torch.empty(batch_size, seq_len, num_splits, num_groups, state_dim, device=device, dtype=torch.float32)
    
    grid = lambda META: (triton.cdiv(chunk_size, META["CHUNK_TILE_SIZE"]) * triton.cdiv(state_dim, META["STATE_TILE_SIZE"]), batch_size * num_chunks, num_splits * num_groups)
    chunk_scan_backward_C_grad_kernel[grid](
        C, h, decay_cumsum, y_grad, decay_grad_from_C, C_grad, length,
        batch_size, seq_len, chunk_size, num_groups, num_heads, head_dim, state_dim, num_heads_per_program,
        *(C.stride()),
        *(h.stride()),
        *(decay_cumsum.stride()),
        *(y_grad.stride()),
        *(decay_grad_from_C.stride()),
        *(C_grad.stride()),
        *(length.stride()),
        HEAD_DIM_ALIGNED=max(triton.next_power_of_2(head_dim), 16)
    )
    C_grad = C_grad.sum(2)
    
    return C_grad, decay_grad_from_C


####################################################################################################


def chunk_scan_backward_CB(
    u: torch.Tensor,
    delta: torch.Tensor,
    decay_cumsum: torch.Tensor,
    y_grad: torch.Tensor,
    length: torch.Tensor,
    num_groups: int
) -> torch.Tensor:
    """
    Args:
        u: (batch_size, seq_len, num_heads, head_dim)
        delta: (batch_size, num_heads, num_chunks, chunk_size)
        decay_cumsum: (batch_size, num_heads, num_chunks, chunk_size)
        y_grad: (batch_size, seq_len, num_heads, head_dim)
        length: (batch_size,)

    Returns:
        CB_grad: (batch_size, num_chunks, num_groups, chunk_size, chunk_size)
    """
    batch_size, seq_len, num_heads, head_dim = u.shape
    _, _, num_chunks, chunk_size = delta.shape
    device = u.device
    num_heads_per_group = num_heads // num_groups
    
    sm_count = torch.cuda.get_device_properties(device).multi_processor_count
    num_heads_per_program = max(min(math.ceil(batch_size * num_chunks * num_heads / sm_count), num_heads_per_group), 1)
    num_splits = triton.cdiv(num_heads_per_group, num_heads_per_program)
    CB_grad = torch.empty(batch_size, num_chunks, num_splits, num_groups, chunk_size, chunk_size, device=device, dtype=torch.float32)
    
    grid = lambda META: (triton.cdiv(chunk_size, META["CHUNK_TILE_Y_SIZE"]) * triton.cdiv(chunk_size, META["CHUNK_TILE_X_SIZE"]), batch_size * num_chunks, num_splits * num_groups)
    chunk_scan_backward_CB_kernel[grid](
        u, delta, decay_cumsum, y_grad, CB_grad, length,
        batch_size, seq_len, chunk_size, num_heads, head_dim, num_groups, num_heads_per_program,
        *(u.stride()),
        *(delta.stride()),
        *(decay_cumsum.stride()),
        *(y_grad.stride()),
        *(CB_grad.stride()),
        *(length.stride()),
        HEAD_DIM_ALIGNED=max(triton.next_power_of_2(head_dim), 16)
    )
    CB_grad = CB_grad.sum(2)

    return CB_grad


####################################################################################################


def bmm_chunk_backward(
    C: torch.Tensor,
    CB_grad: torch.Tensor,
    B_grad: torch.Tensor,
    length: torch.Tensor
) -> torch.Tensor:
    """
    Args:
        C: (batch_size, seq_len, num_groups, state_dim)
        CB_grad: (batch_size, num_chunks, num_groups, chunk_size, chunk_size)
        B_grad: (batch_size, seq_len, num_groups, state_dim)
        length: (batch_size,)

    Returns:
        B_grad: (batch_size, seq_len, num_groups, state_dim)
    """
    batch_size, seq_len, num_groups, state_dim = C.shape
    _, num_chunks, _, _, chunk_size = CB_grad.shape

    grid = lambda META: (triton.cdiv(chunk_size, META["CHUNK_TILE_Y_SIZE"]) * triton.cdiv(state_dim, META["STATE_TILE_SIZE"]), batch_size, num_chunks * num_groups)
    bmm_chunk_backward_kernel[grid](
        C, CB_grad, B_grad, length,
        seq_len, chunk_size, state_dim, num_groups,
        *(C.stride()),
        *(CB_grad.stride()),
        *(B_grad.stride()),
        *(length.stride())
    )

    return B_grad


####################################################################################################

# ->
def chunk_scan_backward_decay_cumsum(
    u: torch.Tensor,
    delta: torch.Tensor,
    decay_cumsum: torch.Tensor,
    y_grad: torch.Tensor,
    CB: torch.Tensor,
    length: torch.Tensor
) -> torch.Tensor:
    """
    Args:
        u: (batch_size, seq_len, num_heads, head_dim)
        delta: (batch_size, num_heads, num_chunks, chunk_size)
        decay_cumsum: (batch_size, num_heads, num_chunks, chunk_size)
        y_grad: (batch_size, seq_len, num_heads, head_dim)
        CB: (batch_size, num_chunks, num_groups, chunk_size, chunk_size)
        length: (batch_size,)
        
    Returns:
        decay_grad: (batch_size, num_heads, num_chunks)
    """
    batch_size, seq_len, num_heads, head_dim = u.shape
    _, _, num_chunks, chunk_size = delta.shape
    num_groups = CB.shape[2]
    device = u.device

    CHUNK_TILE_Y_SIZE_MIN = 32
    decay_grad = torch.empty(batch_size, num_heads, num_chunks, triton.cdiv(chunk_size, CHUNK_TILE_Y_SIZE_MIN), chunk_size, device=device, dtype=torch.float32)

    grid = lambda META: (triton.cdiv(chunk_size, META["CHUNK_TILE_Y_SIZE"]), batch_size * num_chunks, num_heads)
    chunk_scan_backward_decay_cumsum_kernel[grid](
        u, delta, decay_cumsum, y_grad, CB, decay_grad, length,
        batch_size, seq_len, chunk_size, head_dim, num_heads // num_groups,
        *(u.stride()),
        *(delta.stride()),
        *(decay_cumsum.stride()),
        *(y_grad.stride()),
        *(CB.stride()),
        *(decay_grad.stride()),
        *(length.stride()),
        HEAD_DIM_ALIGNED=max(triton.next_power_of_2(head_dim), 16)
    )   
    CHUNK_TILE_Y_SIZE_ACTUAL = chunk_scan_backward_decay_cumsum_kernel.best_config.kwargs["CHUNK_TILE_Y_SIZE"]
    num_valid_tiles = (chunk_size + CHUNK_TILE_Y_SIZE_ACTUAL - 1) // CHUNK_TILE_Y_SIZE_ACTUAL
    decay_grad = decay_grad[:, :, :, :num_valid_tiles].sum(3)

    return decay_grad


####################################################################################################


def chunk_cumsum_backward(
    A: torch.Tensor,
    delta_raw: torch.Tensor,
    delta_bias: torch.Tensor | None,
    delta_grad: torch.Tensor,
    decay_grad_total: torch.Tensor,
    length: torch.Tensor,
    use_delta_softplus: bool,
    delta_limit: Tuple
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """
    Args:
        A: (num_heads,)
        delta_raw: (batch_size, seq_len, num_heads)
        delta_bias: (num_heads,)
        delta_grad: (batch_size, num_heads, num_chunks, chunk_size)
        decay_cumsum_grad: (batch_size, num_heads, num_chunks, chunk_size)
        length: (batch_size,)
        
    Returns:
        A_grad: (num_heads,)
        delta_raw_grad: (batch_size, seq_len, num_heads)
        delta_bias_grad: (num_heads,)
    """
    batch_size, seq_len, num_heads = delta_raw.shape
    _, _, num_chunks, chunk_size = decay_grad_total.shape
    
    delta_bias_grad = torch.empty_like(delta_bias) if delta_bias is not None else None
    delta_raw_grad = torch.empty_like(delta_raw)
    A_grad = torch.empty_like(A)

    grid = lambda META: (batch_size, num_chunks, triton.cdiv(num_heads, META["HEAD_GROUP_SIZE"]))
    chunk_cumsum_backward_kernel[grid](
        A, delta_raw, delta_bias, delta_grad, decay_grad_total, delta_bias_grad, delta_raw_grad, A_grad, length,
        seq_len, num_heads, chunk_size, delta_limit[0], delta_limit[1],
        *(A.stride()),
        *(delta_raw.stride()),
        *(delta_bias.stride() if delta_bias is not None else (0)),
        *(delta_grad.stride()),
        *(decay_grad_total.stride()),
        *(delta_bias_grad.stride() if delta_bias is not None else (0)),
        *(delta_raw_grad.stride()),
        *(A_grad.stride()),
        *(length.stride()),
        USE_DELTA_SOFTPLUS=use_delta_softplus,
        HAS_DELTA_BIAS=delta_bias is not None,
        CHUNK_SIZE_ALIGNED=triton.next_power_of_2(chunk_size)
    )

    return A_grad, delta_raw_grad, delta_bias_grad


####################################################################################################


def ssd_scan_backward(
    u: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    delta_raw: torch.Tensor,
    delta_bias: torch.Tensor | None,
    h_init: torch.Tensor,
    y_grad: torch.Tensor,
    h_last_grad: torch.Tensor,
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
        y_grad: (batch_size, seq_len, num_heads, head_dim)
        h_last_grad: (batch_size, num_heads, head_dim, state_dim)
        length: (batch_size,)
        
    Returns:
        u_grad: (batch_size, seq_len, num_heads, head_dim)  
        A_grad: (num_heads,)
        B_grad, C_grad: (batch_size, seq_len, num_groups, state_dim)
        delta_raw_grad: (batch_size, seq_len, num_heads)
        delta_bias_grad: (num_heads,)
        h_init_grad: (batch_size, num_heads, head_dim, state_dim)
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
    h, _ = state_passing_forward(
        h.view(batch_size, num_chunks, num_heads, -1), 
        h_init.view(batch_size, num_heads, -1) if h_init is not None else None, 
        decay_last
    )
    h = h.view(batch_size, num_chunks, num_heads, head_dim, -1)
    CB = bmm_chunk_forward(B, C, chunk_size, is_causal)
    
    h_grad = chunk_scan_backward_h(C, decay_cumsum, y_grad, length)

    h_grad, decay_last_grad, h_init_grad = state_passing_backward(
        h.view(batch_size, num_chunks, num_heads, head_dim * state_dim),
        decay_last,
        h_grad.view(batch_size, num_chunks, num_heads, head_dim * state_dim),
        h_last_grad.view(batch_size, num_heads, head_dim * state_dim),
        has_h_init=h_init is not None
    )
    h_grad = h_grad.view(batch_size, num_chunks, num_heads, head_dim, state_dim)
    if h_init_grad is not None:
        h_init_grad = h_init_grad.view(batch_size, num_heads, head_dim, state_dim)    
    u_grad, delta_grad = chunk_scan_chunk_state_backward_u(u, delta, decay_cumsum, B, CB, y_grad, h_grad, length)

    B_grad, decay_grad_from_B = chunk_state_backward_B(u, B, delta, decay_cumsum, h_grad, length)

    C_grad, decay_grad_from_C = chunk_scan_backward_C(C, h, decay_cumsum, y_grad, length)

    CB_grad = chunk_scan_backward_CB(u, delta, decay_cumsum, y_grad, length, num_groups=num_groups)

    B_grad = bmm_chunk_backward(C, CB_grad, B_grad, length)
    C_grad = bmm_chunk_backward(B, CB_grad.transpose(-1, -2).contiguous(), C_grad, length)

    decay_grad_from_C[..., -1] += decay_last_grad
    decay_grad_prev_cumsum = decay_grad_from_C.flip([-1]).cumsum(dim=-1).flip([-1])

    decay_grad = chunk_scan_backward_decay_cumsum(u, delta, decay_cumsum, y_grad, CB, length)
    decay_grad_total = decay_grad + decay_grad_from_B + decay_grad_prev_cumsum

    A_grad, delta_raw_grad, delta_bias_grad = chunk_cumsum_backward(
        A, delta_raw, delta_bias, delta_grad, decay_grad_total, length,
        use_delta_softplus=use_delta_softplus, delta_limit=delta_limit
    )

    return u_grad, A_grad, B_grad, C_grad, delta_raw_grad, delta_bias_grad, h_init_grad