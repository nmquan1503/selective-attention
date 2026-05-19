import triton
import triton.language as tl

from ..softplus import softplus

@triton.jit
def ssu_forward_kernel(
    u_ptr,  # (batch_size, num_heads, head_dim)
    A_ptr,  # (num_heads,)
    B_ptr,  # (batch_size, num_groups, state_dim)
    C_ptr,  # (batch_size, num_groups, state_dim)
    delta_raw_ptr,  # (batch_size, num_heads)
    delta_bias_ptr, # (num_heads,)
    h_ptr,  # (batch_size, num_heads, head_dim, state_dim)
    y_ptr,  # (batch_size, num_heads, head_dim)
    batch_size, num_heads, head_dim, state_dim, num_heads_per_group, delta_min, delta_max,
    u_batch_stride, u_head_stride, u_head_element_stride,
    A_head_stride,
    B_batch_stride, B_group_stride, B_state_element_stride,
    C_batch_stride, C_group_stride, C_state_element_stride,
    delta_raw_batch_stride, delta_raw_head_stride,
    delta_bias_head_stride,
    h_batch_stride, h_head_stride, h_head_element_stride, h_state_element_stride,
    y_batch_stride, y_head_stride, y_head_element_stride,
    HAS_DELTA_BIAS: tl.constexpr,
    USE_DELTA_SOFTPLUS: tl.constexpr,
    STATE_DIM_ALIGNED: tl.constexpr,
    HEAD_TILE_SIZE: tl.constexpr
):
    head_tile_id = tl.program_id(axis=0)
    batch_id = tl.program_id(axis=1)
    head_id = tl.program_id(axis=2)
    group_id = head_id // num_heads_per_group

    head_element_ids = head_tile_id * HEAD_TILE_SIZE + tl.arange(0, HEAD_TILE_SIZE)
    state_element_ids = tl.arange(0, STATE_DIM_ALIGNED)

    u_ptr += batch_id * u_batch_stride + head_id * u_head_stride
    A_ptr += head_id * A_head_stride
    B_ptr += batch_id * B_batch_stride + group_id * B_group_stride
    C_ptr += batch_id * C_batch_stride + group_id * C_group_stride
    delta_raw_ptr += batch_id * delta_raw_batch_stride + head_id * delta_raw_head_stride
    if HAS_DELTA_BIAS:
        delta_bias_ptr += head_id * delta_bias_head_stride
    h_ptr += batch_id * h_batch_stride + head_id * h_head_stride
    y_ptr += batch_id * y_batch_stride + head_id * y_head_stride

    u_ptrs = u_ptr + head_element_ids * u_head_element_stride
    B_ptrs = B_ptr + state_element_ids * B_state_element_stride
    C_ptrs = C_ptr + state_element_ids * C_state_element_stride
    h_ptrs = h_ptr + (head_element_ids[:, None] * h_head_element_stride + state_element_ids * h_state_element_stride)
    y_ptrs = y_ptr + head_element_ids * y_head_element_stride

    u = tl.load(u_ptrs)
    A = tl.load(A_ptr)
    B = tl.load(B_ptrs, mask=state_element_ids < state_dim, other=0.0)
    C = tl.load(C_ptrs, mask=state_element_ids < state_dim, other=0.0)
    
    delta_raw = tl.load(delta_raw_ptr)
    if HAS_DELTA_BIAS:
        delta_bias = tl.load(delta_bias_ptr)
        delta = delta_raw + delta_bias
    else:
        delta= delta_raw

    if USE_DELTA_SOFTPLUS:
        delta = tl.where(delta <= 20.0, softplus(delta), delta)
    
    delta = tl.minimum(tl.maximum(delta, delta_min), delta_max)
    
    h = tl.load(h_ptrs, mask=(head_element_ids[:, None] < head_dim) & (state_element_ids[None, :] < state_dim), other=0.0)

    h_new = h * tl.exp(A * delta) + B[None, :] * delta * u[:, None]

    y = tl.sum(h_new * C[None, :], axis=1)

    tl.store(h_ptrs, h_new, mask=(head_element_ids[:, None] < head_dim) & (state_element_ids[None, :] < state_dim))
    tl.store(y_ptrs, y, mask=head_element_ids < head_dim)


