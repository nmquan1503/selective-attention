import triton
import triton.language as tl

from ..hooks import reset_buffers
from ..softplus import softplus


@triton.autotune(
    configs=[
        triton.Config({'HEAD_TILE_SIZE': 128, 'STATE_TILE_SIZE': 256, 'CHUNK_REDUCE_SIZE': 64}, num_stages=3, num_warps=8),
        triton.Config({'HEAD_TILE_SIZE': 64, 'STATE_TILE_SIZE': 256, 'CHUNK_REDUCE_SIZE': 32}, num_stages=4, num_warps=4),
        triton.Config({'HEAD_TILE_SIZE': 128, 'STATE_TILE_SIZE': 128, 'CHUNK_REDUCE_SIZE': 32}, num_stages=4, num_warps=4),
        triton.Config({'HEAD_TILE_SIZE': 128, 'STATE_TILE_SIZE': 64, 'CHUNK_REDUCE_SIZE': 32}, num_stages=4, num_warps=4),
        triton.Config({'HEAD_TILE_SIZE': 64, 'STATE_TILE_SIZE': 128, 'CHUNK_REDUCE_SIZE': 32}, num_stages=4, num_warps=4),
        triton.Config({'HEAD_TILE_SIZE': 128, 'STATE_TILE_SIZE': 32, 'CHUNK_REDUCE_SIZE': 32}, num_stages=4, num_warps=4),
        triton.Config({'HEAD_TILE_SIZE': 64, 'STATE_TILE_SIZE': 32, 'CHUNK_REDUCE_SIZE': 32}, num_stages=5, num_warps=2),
        triton.Config({'HEAD_TILE_SIZE': 32, 'STATE_TILE_SIZE': 64, 'CHUNK_REDUCE_SIZE': 32}, num_stages=5, num_warps=2),
        triton.Config({'HEAD_TILE_SIZE': 64, 'STATE_TILE_SIZE': 64, 'CHUNK_REDUCE_SIZE': 32}, num_stages=4, num_warps=2),
    ],
    key=['head_dim', 'state_dim', 'chunk_size'],
)
@triton.jit
def chunk_scan_backward_h_kernel(
    C_ptr,  # (batch_size, seq_len, num_groups, state_dim)
    decay_cumsum_ptr,   # (batch_size, num_heads, num_chunks, chunk_size)
    y_grad_ptr, # (batch_size, seq_len, num_heads, head_dim)
    h_grad_ptr, # (batch_size, num_chunks, num_heads, head_dim, state_dim)
    length_ptr, # (batch_size)

    batch_size, seq_len, num_chunks, chunk_size, head_dim, state_dim, num_heads_per_group,

    C_batch_stride, C_seq_stride, C_group_stride, C_state_element_stride,
    decay_cumsum_batch_stride, decay_cumsum_head_stride, decay_cumsum_chunk_stride, decay_cumsum_chunk_element_stride,
    y_grad_batch_stride, y_grad_seq_stride, y_grad_head_stride, y_grad_head_element_stride,
    h_grad_batch_stride, h_grad_chunk_stride, h_grad_head_stride, h_grad_head_element_stride, h_grad_state_element_stride,
    length_batch_stride,

    HEAD_TILE_SIZE: tl.constexpr,
    STATE_TILE_SIZE: tl.constexpr,
    CHUNK_REDUCE_SIZE: tl.constexpr
):
    # Map program IDs to batch, chunk, head, and tile positions
    batch_chunk_id = tl.program_id(axis=1)
    chunk_id = batch_chunk_id // batch_size
    batch_id = batch_chunk_id - chunk_id * batch_size
    head_id = tl.program_id(axis=2)
    num_state_tiles = tl.cdiv(state_dim, STATE_TILE_SIZE)
    head_tile_id = tl.program_id(axis=0) // num_state_tiles
    state_tile_id = tl.program_id(axis=0) % num_state_tiles

    # Compute the element indices for this tile
    head_tile_element_ids = head_tile_id * HEAD_TILE_SIZE + tl.arange(0, HEAD_TILE_SIZE)
    state_tile_element_ids = state_tile_id * STATE_TILE_SIZE + tl.arange(0, STATE_TILE_SIZE)
    chunk_reduce_element_ids = tl.arange(0, CHUNK_REDUCE_SIZE)

    # Advance raw pointers to the current batch, chunk, and head
    C_ptr += batch_id * C_batch_stride + chunk_id * chunk_size * C_seq_stride + (head_id // num_heads_per_group) * C_group_stride
    decay_cumsum_ptr += batch_id * decay_cumsum_batch_stride + chunk_id * decay_cumsum_chunk_stride + head_id * decay_cumsum_head_stride
    y_grad_ptr += batch_id * y_grad_batch_stride + chunk_id * chunk_size * y_grad_seq_stride + head_id * y_grad_head_stride
    h_grad_ptr += batch_id * h_grad_batch_stride + chunk_id * h_grad_chunk_stride + head_id * h_grad_head_stride
    length_ptr += batch_id * length_batch_stride

    # Set up matrix pointers for the tile
    C_ptrs = C_ptr + (state_tile_element_ids[None, :] * C_state_element_stride + chunk_reduce_element_ids[:, None] * C_seq_stride)
    decay_cumsum_ptrs = decay_cumsum_ptr + chunk_reduce_element_ids * decay_cumsum_chunk_element_stride
    y_grad_ptrs = y_grad_ptr + (head_tile_element_ids[:, None] * y_grad_head_element_stride + chunk_reduce_element_ids[None, :] * y_grad_seq_stride)
    h_grad_ptrs = h_grad_ptr + (head_tile_element_ids[:, None] * h_grad_head_element_stride + state_tile_element_ids[None, :] * h_grad_state_element_stride)

    length = tl.load(length_ptr)
    chunk_size_limit = min(chunk_size, length - chunk_id * chunk_size)

    # Accumulate h_grad = sum over chunk positions: y_grad * exp(decay_cumsum) * C
    h_grad = tl.zeros((HEAD_TILE_SIZE, STATE_TILE_SIZE), dtype=tl.float32)
    for chunk_reduce_id in range(1 + (chunk_size_limit - 1) // CHUNK_REDUCE_SIZE):
        y_grad = tl.load(
            y_grad_ptrs + chunk_reduce_id * CHUNK_REDUCE_SIZE * y_grad_seq_stride,
            mask=(head_tile_element_ids[:, None] < head_dim) & (chunk_reduce_element_ids[None, :] < chunk_size_limit - chunk_reduce_id * CHUNK_REDUCE_SIZE), other=0.0
        )
        decay_cumsum = tl.load(
            decay_cumsum_ptrs + chunk_reduce_id * CHUNK_REDUCE_SIZE * decay_cumsum_chunk_element_stride,
            mask=chunk_reduce_element_ids < chunk_size - chunk_reduce_id * CHUNK_REDUCE_SIZE, other=0.0
        )
        scale = tl.exp(decay_cumsum)
        y_grad *= scale
        C = tl.load(
            C_ptrs + chunk_reduce_id * CHUNK_REDUCE_SIZE * C_seq_stride,
            mask=(chunk_reduce_element_ids[:, None] < chunk_size_limit - chunk_reduce_id * CHUNK_REDUCE_SIZE) & (state_tile_element_ids[None, :] < state_dim), other=0.0
        )
        h_grad += tl.dot(y_grad, C)
    
    # Store accumulated h_grad
    tl.store(h_grad_ptrs, h_grad, mask=(head_tile_element_ids[:, None] < head_dim) & (state_tile_element_ids[None, :] < state_dim))


####################################################################################################


@triton.autotune(
    configs=[
        triton.Config({'HEAD_STATE_GROUP_SIZE': 64}),
        triton.Config({'HEAD_STATE_GROUP_SIZE': 128}),
        triton.Config({'HEAD_STATE_GROUP_SIZE': 256}),
        triton.Config({'HEAD_STATE_GROUP_SIZE': 512}),
        triton.Config({'HEAD_STATE_GROUP_SIZE': 1024}),
        triton.Config({'HEAD_STATE_GROUP_SIZE': 2048}),
    ],
    key=['head_state_dim'],
)
@triton.jit
def state_passing_backward_kernel(
    h_ptr,  # (batch_size, num_chunks, num_heads, head_dim * state_dim)
    decay_last_ptr, # (batch_size, num_heads, num_chunks)
    h_grad_ptr, # (batch_size, num_chunks, num_heads, head_dim * state_dim)
    h_last_grad_ptr,    # (batch_size, num_heads, head_dim * state_dim),
    decay_last_grad_ptr,    # (batch_size, num_heads, num_chunks, num_head_state_groups)
    h_init_grad_ptr,    # (batch_size, num_heads, head_dim * state_dim)

    num_chunks, head_state_dim,

    h_batch_stride, h_chunk_stride, h_head_stride, h_head_state_element_stride,
    decay_last_batch_stride, decay_last_head_stride, decay_last_chunk_stride,
    h_grad_batch_stride, h_grad_chunk_stride, h_grad_head_stride, h_grad_head_state_element_stride,
    h_last_grad_batch_stride, h_last_grad_head_stride, h_last_grad_head_state_element_stride,
    decay_last_grad_batch_stride, decay_last_grad_head_stride, decay_last_grad_chunk_stride, decay_last_grad_head_state_group_stride,
    h_init_grad_batch_stride, h_init_grad_head_stride, h_init_grad_head_state_element_stride,
    
    HAS_H_INIT: tl.constexpr,
    HEAD_STATE_GROUP_SIZE: tl.constexpr
):
    # Map program IDs to batch, head, and head-state group
    batch_id = tl.program_id(axis=1)
    head_id = tl.program_id(axis=2)
    head_state_group_id = tl.program_id(axis=0)

    head_state_element_ids = head_state_group_id * HEAD_STATE_GROUP_SIZE + tl.arange(0, HEAD_STATE_GROUP_SIZE)
    mask = head_state_element_ids < head_state_dim

    # Advance pointers to current batch and head
    h_ptr += batch_id * h_batch_stride + head_id * h_head_stride
    h_grad_ptr += batch_id * h_grad_batch_stride + head_id * h_grad_head_stride
    decay_last_ptr += batch_id * decay_last_batch_stride + head_id * decay_last_head_stride
    decay_last_grad_ptr += batch_id * decay_last_grad_batch_stride + head_id * decay_last_grad_head_stride + head_state_group_id * decay_last_grad_head_state_group_stride
    h_last_grad_ptr += batch_id * h_last_grad_batch_stride + head_id * h_last_grad_head_stride
    if HAS_H_INIT:
        h_init_grad_ptr += batch_id * h_init_grad_batch_stride + head_id * h_init_grad_head_stride
    
    h_grad_ptrs = h_grad_ptr + head_state_element_ids * h_grad_head_state_element_stride
    h_ptrs = h_ptr + head_state_element_ids * h_head_state_element_stride

    # Backward recurrence: propagate gradient from last chunk to first
    h_next_grad = tl.load(h_last_grad_ptr + head_state_element_ids * h_last_grad_head_state_element_stride, mask=mask, other=0.0)
    for chunk_id in range(num_chunks - 1, 0, -1):
        scale = tl.exp(tl.load(decay_last_ptr + chunk_id * decay_last_chunk_stride).to(tl.float32))
        h = tl.load(h_ptrs + chunk_id * h_chunk_stride, mask=mask, other=0.0).to(tl.float32)
        # Gradient for decay_last: sum(h * h_next_grad) * scale
        decay_last_grad = tl.sum(h * h_next_grad) * scale
        tl.store(decay_last_grad_ptr + chunk_id * decay_last_grad_chunk_stride, decay_last_grad)
        # Combine intra-chunk gradient with the gradient propagated from the right
        h_grad_intra = tl.load(h_grad_ptrs + chunk_id * h_grad_chunk_stride, mask=mask, other=0.0).to(tl.float32)
        tl.store(h_grad_ptrs + chunk_id * h_grad_chunk_stride, h_next_grad, mask=mask)
        h_next_grad = scale * h_next_grad + h_grad_intra

    # Handle the first chunk (chunk 0) and h_init_grad if present
    scale0 = tl.exp(tl.load(decay_last_ptr + 0 * decay_last_chunk_stride).to(tl.float32))
    h0 = tl.load(h_ptrs + 0 * h_chunk_stride, mask=mask, other=0.0).to(tl.float32)
    chunk0_grad = tl.sum(h0 * h_next_grad) * scale0
    tl.store(decay_last_grad_ptr + 0 * decay_last_grad_chunk_stride, chunk0_grad)
    h_grad0_intra = tl.load(h_grad_ptrs + 0 * h_grad_chunk_stride, mask=mask, other=0.0).to(tl.float32)
    tl.store(h_grad_ptrs + 0 * h_grad_chunk_stride, h_next_grad, mask=mask)
    
    if HAS_H_INIT:
        h_init_grad = scale0 * h_next_grad + h_grad0_intra
        tl.store(h_init_grad_ptr + head_state_element_ids * h_init_grad_head_state_element_stride, h_init_grad, mask=mask)
    

####################################################################################################


@triton.autotune(
    configs=[
        triton.Config({'CHUNK_TILE_SIZE': 128, 'HEAD_TILE_SIZE': 256, 'REDUCE_SIZE': 64}, num_stages=3, num_warps=8, pre_hook=reset_buffers(["delta_grad_ptr"])),
        triton.Config({'CHUNK_TILE_SIZE': 64, 'HEAD_TILE_SIZE': 256, 'REDUCE_SIZE': 32}, num_stages=4, num_warps=4, pre_hook=reset_buffers(["delta_grad_ptr"])),
        triton.Config({'CHUNK_TILE_SIZE': 128, 'HEAD_TILE_SIZE': 128, 'REDUCE_SIZE': 32}, num_stages=4, num_warps=4, pre_hook=reset_buffers(["delta_grad_ptr"])),
        triton.Config({'CHUNK_TILE_SIZE': 128, 'HEAD_TILE_SIZE': 64, 'REDUCE_SIZE': 32}, num_stages=4, num_warps=4, pre_hook=reset_buffers(["delta_grad_ptr"])),
        triton.Config({'CHUNK_TILE_SIZE': 64, 'HEAD_TILE_SIZE': 128, 'REDUCE_SIZE': 32}, num_stages=4, num_warps=4, pre_hook=reset_buffers(["delta_grad_ptr"])),
        triton.Config({'CHUNK_TILE_SIZE': 128, 'HEAD_TILE_SIZE': 32, 'REDUCE_SIZE': 32}, num_stages=4, num_warps=4, pre_hook=reset_buffers(["delta_grad_ptr"])),
        triton.Config({'CHUNK_TILE_SIZE': 64, 'HEAD_TILE_SIZE': 32, 'REDUCE_SIZE': 32}, num_stages=5, num_warps=4, pre_hook=reset_buffers(["delta_grad_ptr"])),
        triton.Config({'CHUNK_TILE_SIZE': 32, 'HEAD_TILE_SIZE': 64, 'REDUCE_SIZE': 32}, num_stages=5, num_warps=4, pre_hook=reset_buffers(["delta_grad_ptr"])),
        triton.Config({'CHUNK_TILE_SIZE': 64, 'HEAD_TILE_SIZE': 64, 'REDUCE_SIZE': 32}, num_stages=4, num_warps=4, pre_hook=reset_buffers(["delta_grad_ptr"])),
    ],
    key=['chunk_size', 'head_dim', 'state_dim'],
)
@triton.jit
def chunk_scan_chunk_state_backward_u_kernel(
    u_ptr,  # (batch_size, seq_len, num_heads, head_dim)
    delta_ptr,  # (batch_size, num_heads, num_chunks, chunk_size)
    decay_cumsum_ptr, # (batch_size, num_heads, num_chunks, chunk_size)
    B_ptr,  # (batch_size, seq_len, num_groups, state_dim)
    CB_ptr, # (batch_size, num_chunks, num_groups, chunk_size, chunk_size)
    y_grad_ptr, # (batch_size, seq_len, num_heads, head_dim)
    h_grad_ptr, # (batch_size, num_chunks, num_heads, head_dim, state_dim)
    u_grad_ptr, # (batch_size, seq_len, num_heads, head_dim)
    delta_grad_ptr, # (batch_size, num_heads, num_chunks, chunk_size)
    length_ptr, # (batch_size,)

    batch_size, seq_len, chunk_size, head_dim, state_dim, num_heads_per_group,

    u_batch_stride, u_seq_stride, u_head_stride, u_head_element_stride,
    delta_batch_stride, delta_head_stride, delta_chunk_stride, delta_chunk_element_stride,
    decay_cumsum_batch_stride, decay_cumsum_head_stride, decay_cumsum_chunk_stride, decay_chunk_element_stride,
    B_batch_stride, B_seq_stride, B_group_stride, B_state_element_stride,
    CB_batch_stride, CB_chunk_stride, CB_group_stride, CB_chunk_y_element_stride, CB_chunk_x_element_stride,
    y_grad_batch_stride, y_grad_seq_stride, y_grad_head_stride, y_grad_head_element_stride,
    h_grad_batch_stride, h_grad_chunk_stride, h_grad_head_stride, h_grad_head_element_stride, h_grad_state_element_stride,
    u_grad_batch_stride, u_grad_seq_stride, u_grad_head_stride, u_grad_head_element_stride,
    delta_grad_batch_stride, delta_grad_head_stride, delta_grad_chunk_stride, delta_grad_chunk_element_stride,
    length_batch_stride,

    CHUNK_TILE_SIZE: tl.constexpr,
    HEAD_TILE_SIZE: tl.constexpr,
    REDUCE_SIZE: tl.constexpr,
    STATE_DIM_ALIGNED: tl.constexpr,
):
    # Map program IDs to batch, chunk, head, and tile positions
    batch_chunk_id = tl.program_id(axis=1)
    chunk_id = batch_chunk_id // batch_size
    batch_id = batch_chunk_id - chunk_id * batch_size
    head_id = tl.program_id(axis=2)
    num_head_tiles = tl.cdiv(head_dim, HEAD_TILE_SIZE)
    chunk_tile_id = tl.program_id(axis=0) // num_head_tiles
    head_tile_id = tl.program_id(axis=0) % num_head_tiles
    group_id = head_id // num_heads_per_group

    # Compute element indices for this tile
    chunk_tile_element_ids = chunk_tile_id * CHUNK_TILE_SIZE + tl.arange(0, CHUNK_TILE_SIZE)
    head_tile_element_ids = head_tile_id * HEAD_TILE_SIZE + tl.arange(0, HEAD_TILE_SIZE)
    state_reduce_element_ids = tl.arange(0, STATE_DIM_ALIGNED if STATE_DIM_ALIGNED <= 128 else REDUCE_SIZE)
    chunk_reduce_element_ids = tl.arange(0, REDUCE_SIZE)

    # Advance raw pointers to the current batch, chunk, and head
    u_ptr += batch_id * u_batch_stride + chunk_id * chunk_size * u_seq_stride + head_id * u_head_stride
    delta_ptr += batch_id * delta_batch_stride + head_id * delta_head_stride + chunk_id * delta_chunk_stride
    decay_cumsum_ptr += batch_id * decay_cumsum_batch_stride + head_id * decay_cumsum_head_stride + chunk_id * decay_cumsum_chunk_stride
    B_ptr += batch_id * B_batch_stride + chunk_id * chunk_size * B_seq_stride + group_id * B_group_stride
    CB_ptr += batch_id * CB_batch_stride + chunk_id * CB_chunk_stride + group_id * CB_group_stride
    y_grad_ptr += batch_id * y_grad_batch_stride + chunk_id * chunk_size * y_grad_seq_stride + head_id * y_grad_head_stride
    h_grad_ptr += batch_id * h_grad_batch_stride + chunk_id * h_grad_chunk_stride + head_id * h_grad_head_stride
    u_grad_ptr += batch_id * u_grad_batch_stride + chunk_id * chunk_size * u_grad_seq_stride + head_id * u_grad_head_stride
    delta_grad_ptr += batch_id * delta_grad_batch_stride + head_id * delta_grad_head_stride + chunk_id * delta_grad_chunk_stride
    length_ptr += batch_id * length_batch_stride

    length = tl.load(length_ptr)
    chunk_size_limit = min(chunk_size, length - chunk_id * chunk_size)
    acc = tl.zeros((CHUNK_TILE_SIZE, HEAD_TILE_SIZE), dtype=tl.float32)

    # Precompute scale = exp(decay_cumsum_last - decay_cumsum_current) for the chunk tile
    decay_cumsum_ptrs = decay_cumsum_ptr + chunk_tile_element_ids * decay_chunk_element_stride
    decay_cumsum_chunk_tile = tl.load(decay_cumsum_ptrs, mask=chunk_tile_element_ids < chunk_size_limit, other=0.0)
    decay_cumsum_last = tl.load(decay_cumsum_ptr + (chunk_size - 1) * decay_chunk_element_stride)
    scale = tl.exp(tl.minimum(decay_cumsum_last - decay_cumsum_chunk_tile, 0.0))

    # Accumulate contribution from B @ h_grad
    B_ptrs = B_ptr + (chunk_tile_element_ids[:, None] * B_seq_stride + state_reduce_element_ids[None, :] * B_state_element_stride)
    h_grad_ptrs = h_grad_ptr + (head_tile_element_ids[None, :] * h_grad_head_element_stride + state_reduce_element_ids[:, None] * h_grad_state_element_stride)
    for state_offset in range(0, state_dim, REDUCE_SIZE):
        B = tl.load(B_ptrs, mask=(chunk_tile_element_ids[:, None] < chunk_size_limit) & (state_reduce_element_ids[None, :] < state_dim - state_offset), other=0.0)
        h_grad = tl.load(h_grad_ptrs, mask=(state_reduce_element_ids[:, None] < state_dim - state_offset) & (head_tile_element_ids[None, :] < head_dim), other=0.0)
        acc += tl.dot(B, h_grad)
        B_ptrs += REDUCE_SIZE * B_state_element_stride
        h_grad_ptrs += REDUCE_SIZE * h_grad_state_element_stride

    acc *= scale[:, None]

    # Accumulate contribution from CB @ y_grad, with causal masking
    CB_ptrs = CB_ptr + (chunk_tile_element_ids[:, None] * CB_chunk_x_element_stride + chunk_reduce_element_ids[None, :] * CB_chunk_y_element_stride)
    y_grad_ptrs = y_grad_ptr + (chunk_reduce_element_ids[:, None] * y_grad_seq_stride + head_tile_element_ids[None, :] * y_grad_head_element_stride)
    decay_cumsum_reduce_ptrs = decay_cumsum_ptr + chunk_reduce_element_ids * decay_chunk_element_stride

    chunk_end = chunk_size_limit
    chunk_start = chunk_tile_id * CHUNK_TILE_SIZE
    CB_ptrs += chunk_start * CB_chunk_y_element_stride
    y_grad_ptrs += chunk_start * y_grad_seq_stride
    decay_cumsum_reduce_ptrs += chunk_start * decay_chunk_element_stride

    for current_chunk_idx in range(chunk_start, chunk_end, REDUCE_SIZE):
        current_chunk_idx = tl.multiple_of(current_chunk_idx, REDUCE_SIZE)
        CB = tl.load(CB_ptrs, mask=(chunk_tile_element_ids[:, None] < chunk_size) & (chunk_reduce_element_ids[None, :] < chunk_end - current_chunk_idx), other=0.0)
        y_grad = tl.load(y_grad_ptrs, mask=(chunk_reduce_element_ids[:, None] < chunk_end - current_chunk_idx) & (head_tile_element_ids[None, :] < head_dim), other=0.0)
        decay_cumsum_current = tl.load(decay_cumsum_reduce_ptrs, mask=chunk_reduce_element_ids < chunk_end - current_chunk_idx, other=0.0)

        CB *= tl.exp(tl.minimum(decay_cumsum_current[None, :] - decay_cumsum_chunk_tile[:, None], 0.0))

        # Apply lower-triangular mask (column index >= row index)
        triangular_mask = (current_chunk_idx + chunk_reduce_element_ids[None, :] >= chunk_tile_element_ids[:, None]) & (current_chunk_idx + chunk_reduce_element_ids[None, :] < chunk_end)
        CB = tl.where(triangular_mask, CB, 0.0)

        acc += tl.dot(CB, y_grad)

        CB_ptrs += REDUCE_SIZE * CB_chunk_y_element_stride
        y_grad_ptrs += REDUCE_SIZE * y_grad_seq_stride
        decay_cumsum_reduce_ptrs += REDUCE_SIZE * decay_chunk_element_stride

    # Compute u_grad and store
    delta_ptrs = delta_ptr + chunk_tile_element_ids * delta_chunk_element_stride
    delta_chunk_tile = tl.load(delta_ptrs, mask=chunk_tile_element_ids < chunk_size_limit, other=0.0)
    u_grad = acc * delta_chunk_tile[:, None]
    u_grad_ptrs = u_grad_ptr + (chunk_tile_element_ids[:, None] * u_grad_seq_stride + head_tile_element_ids[None, :] * u_grad_head_element_stride)
    tl.store(u_grad_ptrs, u_grad, mask=(chunk_tile_element_ids[:, None] < chunk_size_limit) & (head_tile_element_ids[None, :] < head_dim))

    # Compute delta_grad via atomic add: sum(acc * u, axis=1)
    u_ptrs = u_ptr + (chunk_tile_element_ids[:, None] * u_seq_stride + head_tile_element_ids[None, :] * u_head_element_stride)
    u = tl.load(u_ptrs, mask=(chunk_tile_element_ids[:, None] < chunk_size_limit) & (head_tile_element_ids[None, :] < head_dim), other=0.0)

    delta_grad = tl.sum(acc * u, axis=1)
    delta_grad_ptrs = delta_grad_ptr + chunk_tile_element_ids * delta_grad_chunk_element_stride
    tl.atomic_add(delta_grad_ptrs, delta_grad, mask=chunk_tile_element_ids < chunk_size)


####################################################################################################


@triton.autotune(
    configs=[
        triton.Config({'CHUNK_TILE_SIZE': 32, 'STATE_TILE_SIZE': 128}, num_stages=3, num_warps=4, pre_hook=reset_buffers(["decay_grad_from_B_ptr"])),
        triton.Config({'CHUNK_TILE_SIZE': 128, 'STATE_TILE_SIZE': 32}, num_stages=3, num_warps=4, pre_hook=reset_buffers(["decay_grad_from_B_ptr"])),
        triton.Config({'CHUNK_TILE_SIZE': 64, 'STATE_TILE_SIZE': 128}, num_stages=3, num_warps=4, pre_hook=reset_buffers(["decay_grad_from_B_ptr"])),
        triton.Config({'CHUNK_TILE_SIZE': 128, 'STATE_TILE_SIZE': 64}, num_stages=3, num_warps=4, pre_hook=reset_buffers(["decay_grad_from_B_ptr"])),
        triton.Config({'CHUNK_TILE_SIZE': 64, 'STATE_TILE_SIZE': 64}, num_stages=3, num_warps=4, pre_hook=reset_buffers(["decay_grad_from_B_ptr"])),
        triton.Config({'CHUNK_TILE_SIZE': 64, 'STATE_TILE_SIZE': 32}, num_stages=3, num_warps=4, pre_hook=reset_buffers(["decay_grad_from_B_ptr"])),
        triton.Config({'CHUNK_TILE_SIZE': 32, 'STATE_TILE_SIZE': 64}, num_stages=3, num_warps=4, pre_hook=reset_buffers(["decay_grad_from_B_ptr"])),
        triton.Config({'CHUNK_TILE_SIZE': 32, 'STATE_TILE_SIZE': 32}, num_stages=3, num_warps=4, pre_hook=reset_buffers(["decay_grad_from_B_ptr"])),
    ],
    key=['chunk_size', 'state_dim', 'head_dim'],
)
@triton.jit
def chunk_state_backward_B_kernel(
    u_ptr,  # (batch_size, seq_len, num_heads, head_dim)
    B_ptr,  # (batch_size, seq_len, num_groups, state_dim)
    delta_ptr,  # (batch_size, num_heads, num_chunks, chunk_size)
    decay_cumsum_ptr,   # (batch_size, num_heads, num_chunks, chunk_size)
    h_grad_ptr, # (batch_size, num_chunks, num_heads, head_dim, state_dim)
    decay_grad_from_B_ptr,  # (batch_size, num_heads, num_chunks, chunk_size)
    B_grad_ptr, # (batch_size, seq_len, num_splits, num_groups, state_dim)
    length_ptr, # (batch_size,)

    batch_size, seq_len, chunk_size, num_heads, head_dim, state_dim, num_groups, num_heads_per_program,

    u_batch_stride, u_seq_stride, u_head_stride, u_head_element_stride,
    B_batch_stride, B_seq_stride, B_group_stride, B_state_element_stride,
    delta_batch_stride, delta_head_stride, delta_chunk_stride, delta_chunk_element_stride,
    decay_cumsum_batch_stride, decay_cumsum_head_stride, decay_cumsum_chunk_stride, decay_cumsum_chunk_element_stride,
    h_grad_batch_stride, h_grad_chunk_stride, h_grad_head_stride, h_grad_head_element_stride, h_grad_state_element_stride,
    decay_grad_from_B_batch_stride, decay_grad_from_B_head_stride, decay_grad_from_B_chunk_stride, decay_grad_from_B_chunk_element_stride,
    B_grad_batch_stride, B_grad_seq_stride, B_grad_split_stride, B_grad_group_stride, B_grad_state_dim,
    length_batch_stride,

    CHUNK_TILE_SIZE: tl.constexpr,
    STATE_TILE_SIZE: tl.constexpr,
    HEAD_DIM_ALIGNED: tl.constexpr
):
    # Map program IDs to batch, chunk, split/group, and tile positions
    batch_chunk_id = tl.program_id(axis=1)
    chunk_id = batch_chunk_id // batch_size
    batch_id = batch_chunk_id - chunk_id * batch_size
    split_group_id = tl.program_id(axis=2)
    split_id = split_group_id // num_groups
    group_id = split_group_id - split_id * num_groups
    num_state_tiles = tl.cdiv(state_dim, STATE_TILE_SIZE)
    chunk_tile_id = tl.program_id(axis=0) // num_state_tiles
    state_tile_id = tl.program_id(axis=0) % num_state_tiles

    chunk_tile_element_ids = chunk_tile_id * CHUNK_TILE_SIZE + tl.arange(0, CHUNK_TILE_SIZE)
    state_tile_element_ids = state_tile_id * STATE_TILE_SIZE + tl.arange(0, STATE_TILE_SIZE)
    head_element_ids = tl.arange(0, HEAD_DIM_ALIGNED)
    
    # Advance pointers to the correct batch, chunk, and head group
    head_start = group_id * (num_heads // num_groups) + split_id * num_heads_per_program

    u_ptr += batch_id * u_batch_stride + chunk_id * chunk_size * u_seq_stride + head_start * u_head_stride
    B_grad_ptr += batch_id * B_grad_batch_stride + chunk_id * chunk_size * B_grad_seq_stride + group_id * B_grad_group_stride + split_id * B_grad_split_stride
    h_grad_ptr += batch_id * h_grad_batch_stride + chunk_id * h_grad_chunk_stride + head_start * h_grad_head_stride
    delta_ptr += batch_id * delta_batch_stride + head_start * delta_head_stride + chunk_id * delta_chunk_stride
    decay_cumsum_ptr += batch_id * decay_cumsum_batch_stride + head_start * decay_cumsum_head_stride + chunk_id * decay_cumsum_chunk_stride
    B_ptr += batch_id * B_batch_stride + chunk_id * chunk_size * B_seq_stride + group_id * B_group_stride
    decay_grad_from_B_ptr += batch_id * decay_grad_from_B_batch_stride + head_start * decay_grad_from_B_head_stride + chunk_id * decay_grad_from_B_chunk_stride
    length_ptr += batch_id * length_batch_stride

    length = tl.load(length_ptr)
    chunk_size_limit = min(chunk_size, length - chunk_id * chunk_size)

    B_grad_acc = tl.zeros((CHUNK_TILE_SIZE, STATE_TILE_SIZE), dtype=tl.float32)

    # Pre-load B tile for decay gradient computation
    B_ptrs = B_ptr + (chunk_tile_element_ids[:, None] * B_seq_stride + state_tile_element_ids[None, :] * B_state_element_stride)
    B_tile = tl.load(B_ptrs, mask=(chunk_tile_element_ids[:, None] < chunk_size_limit) & (state_tile_element_ids[None, :] < state_dim), other=0.0)

    # Pointers that advance across heads
    u_ptrs = u_ptr + (chunk_tile_element_ids[:, None] * u_seq_stride + head_element_ids[None, :] * u_head_element_stride)
    h_grad_ptrs = h_grad_ptr + (state_tile_element_ids[None, :] * h_grad_state_element_stride + head_element_ids[:, None] * h_grad_head_element_stride)
    delta_ptrs = delta_ptr + chunk_tile_element_ids * delta_chunk_element_stride
    decay_cumsum_ptrs = decay_cumsum_ptr + chunk_tile_element_ids * decay_cumsum_chunk_element_stride
    decay_grad_from_B_ptrs = decay_grad_from_B_ptr + chunk_tile_element_ids * decay_grad_from_B_chunk_element_stride

    num_heads_iter = min(num_heads_per_program, num_heads // num_groups - split_id * num_heads_per_program)

    # Accumulate B_grad = u @ h_grad, weighted by delta and decay scale
    for _ in range(num_heads_iter):
        u_tile = tl.load(u_ptrs, mask=(chunk_tile_element_ids[:, None] < chunk_size_limit) & (head_element_ids[None, :] < head_dim), other=0.0)
        h_grad_tile = tl.load(h_grad_ptrs, mask=(head_element_ids[:, None] < head_dim) & (state_tile_element_ids[None, :] < state_dim), other=0.0)

        contribution = tl.dot(u_tile, h_grad_tile)

        decay_cumsum_last = tl.load(decay_cumsum_ptr + (chunk_size - 1) * decay_cumsum_chunk_element_stride)

        decay_cumsum_current = tl.load(decay_cumsum_ptrs, mask=chunk_tile_element_ids < chunk_size_limit, other=0.0)
        delta_current = tl.load(delta_ptrs, mask=chunk_tile_element_ids < chunk_size_limit, other=0.0)

        scale = tl.exp(tl.minimum(decay_cumsum_last - decay_cumsum_current, 0.0))

        contribution *= (scale * delta_current)[:, None]

        B_grad_acc += contribution

        # Gradient for decay_cumsum from B path (exclusive cumsum, offset by 1)
        decay_grad_from_B_contrib = tl.sum(contribution * B_tile, axis=1) 
        tl.atomic_add(
            decay_grad_from_B_ptrs + decay_grad_from_B_chunk_element_stride, decay_grad_from_B_contrib,
            mask=chunk_tile_element_ids < chunk_size_limit - 1
        )

        u_ptrs += u_head_stride
        h_grad_ptrs += h_grad_head_stride
        delta_ptrs += delta_head_stride
        decay_cumsum_ptrs += decay_cumsum_head_stride
        decay_grad_from_B_ptrs += decay_grad_from_B_head_stride

    # Store accumulated B_grad for this split
    B_grad_ptrs = B_grad_ptr + (chunk_tile_element_ids[:, None] * B_grad_seq_stride + state_tile_element_ids[None, :] * B_grad_state_dim)
    tl.store(B_grad_ptrs, B_grad_acc, mask=(chunk_tile_element_ids[:, None] < chunk_size_limit) & (state_tile_element_ids[None, :] < state_dim))


####################################################################################################


@triton.autotune(
    configs=[
        triton.Config({'CHUNK_TILE_SIZE': 32, 'STATE_TILE_SIZE': 128}, num_stages=3, num_warps=4, pre_hook=reset_buffers(["decay_grad_from_C_ptr"])),
        triton.Config({'CHUNK_TILE_SIZE': 128, 'STATE_TILE_SIZE': 32}, num_stages=3, num_warps=4, pre_hook=reset_buffers(["decay_grad_from_C_ptr"])),
        triton.Config({'CHUNK_TILE_SIZE': 64, 'STATE_TILE_SIZE': 128}, num_stages=3, num_warps=4, pre_hook=reset_buffers(["decay_grad_from_C_ptr"])),
        triton.Config({'CHUNK_TILE_SIZE': 128, 'STATE_TILE_SIZE': 64}, num_stages=3, num_warps=4, pre_hook=reset_buffers(["decay_grad_from_C_ptr"])),
        triton.Config({'CHUNK_TILE_SIZE': 64, 'STATE_TILE_SIZE': 64}, num_stages=3, num_warps=4, pre_hook=reset_buffers(["decay_grad_from_C_ptr"])),
        triton.Config({'CHUNK_TILE_SIZE': 64, 'STATE_TILE_SIZE': 32}, num_stages=3, num_warps=4, pre_hook=reset_buffers(["decay_grad_from_C_ptr"])),
        triton.Config({'CHUNK_TILE_SIZE': 32, 'STATE_TILE_SIZE': 64}, num_stages=3, num_warps=4, pre_hook=reset_buffers(["decay_grad_from_C_ptr"])),
        triton.Config({'CHUNK_TILE_SIZE': 32, 'STATE_TILE_SIZE': 32}, num_stages=3, num_warps=4, pre_hook=reset_buffers(["decay_grad_from_C_ptr"])),
    ],
    key=['chunk_size', 'state_dim', 'head_dim'],
)
@triton.jit
def chunk_scan_backward_C_grad_kernel(
    C_ptr,  # (batch_size, seq_len, num_groups, state_dim)
    h_ptr,  # (batch_size, num_chunks, num_heads, head_dim, state_dim),
    decay_cumsum_ptr,   # (batch_size, num_heads, num_chunks, chunk_size)
    y_grad_ptr, # (batch_size, seq_len, num_heads, head_dim)
    decay_grad_from_C_ptr,  # (batch_size, num_heads, num_chunks, chunk_size)
    C_grad_ptr, # (batch_size, seq_len, num_splits, num_groups, state_dim)
    length_ptr, # (batch_size,)

    batch_size, seq_len, chunk_size, num_groups, num_heads, head_dim, state_dim, num_heads_per_program,

    C_batch_stride, C_seq_stride, C_group_stride, C_state_element_stride,
    h_batch_stride, h_chunk_stride, h_head_stride, h_head_element_stride, h_state_element_stride,
    decay_cumsum_batch_stride, decay_cumsum_head_stride, decay_cumsum_chunk_stride, decay_cumsum_chunk_element_stride,
    y_grad_batch_stride, y_grad_seq_stride, y_grad_head_stride, y_grad_head_element_stride,
    decay_grad_from_C_batch_stride, decay_grad_from_C_head_stride, decay_grad_from_C_chunk_stride, decay_grad_from_C_chunk_element_stride,
    C_grad_batch_stride, C_grad_seq_stride, C_grad_split_stride, C_grad_group_stride, C_grad_state_element_stride,
    length_batch_stride,

    CHUNK_TILE_SIZE: tl.constexpr,
    STATE_TILE_SIZE: tl.constexpr,
    HEAD_DIM_ALIGNED: tl.constexpr
):
    # Map program IDs to batch, chunk, split/group, and tile positions
    batch_chunk_id = tl.program_id(axis=1)
    chunk_id = batch_chunk_id // batch_size
    batch_id = batch_chunk_id - chunk_id * batch_size
    split_group_id = tl.program_id(axis=2)
    split_id = split_group_id // num_groups
    group_id = split_group_id - split_id * num_groups
    num_state_tiles = tl.cdiv(state_dim, STATE_TILE_SIZE)
    chunk_tile_id = tl.program_id(axis=0) // num_state_tiles
    state_tile_id = tl.program_id(axis=0) % num_state_tiles

    chunk_tile_element_ids = chunk_tile_id * CHUNK_TILE_SIZE + tl.arange(0, CHUNK_TILE_SIZE)
    state_tile_element_ids = state_tile_id * STATE_TILE_SIZE + tl.arange(0, STATE_TILE_SIZE)
    head_element_ids = tl.arange(0, HEAD_DIM_ALIGNED)
    
    # Advance pointers to the correct batch, chunk, and head group
    head_start = group_id * (num_heads // num_groups) + split_id * num_heads_per_program

    y_grad_ptr += batch_id * y_grad_batch_stride + chunk_id * chunk_size * y_grad_seq_stride + head_start * y_grad_head_stride
    C_grad_ptr += batch_id * C_grad_batch_stride + chunk_id * chunk_size * C_grad_seq_stride + group_id * C_grad_group_stride + split_id * C_grad_split_stride
    h_ptr += batch_id * h_batch_stride + chunk_id * h_chunk_stride + head_start * h_head_stride
    decay_cumsum_ptr += batch_id * decay_cumsum_batch_stride + head_start * decay_cumsum_head_stride + chunk_id * decay_cumsum_chunk_stride
    C_ptr += batch_id * C_batch_stride + chunk_id * chunk_size * C_seq_stride + group_id * C_group_stride
    decay_grad_from_C_ptr += batch_id * decay_grad_from_C_batch_stride + head_start * decay_grad_from_C_head_stride + chunk_id * decay_grad_from_C_chunk_stride
    length_ptr += batch_id * length_batch_stride

    length = tl.load(length_ptr)
    chunk_size_limit = min(chunk_size, length - chunk_id * chunk_size)
    C_grad_acc = tl.zeros((CHUNK_TILE_SIZE, STATE_TILE_SIZE), dtype=tl.float32)

    # Pre-load C tile for decay gradient computation
    C_ptrs = C_ptr + (chunk_tile_element_ids[:, None] * C_seq_stride + state_tile_element_ids[None, :] * C_state_element_stride)
    C_tile = tl.load(C_ptrs, mask=(chunk_tile_element_ids[:, None] < chunk_size_limit) & (state_tile_element_ids[None, :] < state_dim), other=0.0)

    # Pointers that advance across heads
    y_grad_ptrs = y_grad_ptr + (chunk_tile_element_ids[:, None] * y_grad_seq_stride + head_element_ids[None, :] * y_grad_head_element_stride)
    h_ptrs = h_ptr + (state_tile_element_ids[None, :] * h_state_element_stride + head_element_ids[:, None] * h_head_element_stride)
    decay_cumsum_ptrs = decay_cumsum_ptr + chunk_tile_element_ids * decay_cumsum_chunk_element_stride
    decay_grad_from_C_ptrs = decay_grad_from_C_ptr + chunk_tile_element_ids * decay_grad_from_C_chunk_element_stride

    num_heads_iter = min(num_heads_per_program, num_heads // num_groups - split_id * num_heads_per_program)
    
    # Accumulate C_grad = y_grad @ h, weighted by exp(decay_cumsum)
    for _ in range(num_heads_iter):
        y_grad_tile = tl.load(y_grad_ptrs, mask=(chunk_tile_element_ids[:, None] < chunk_size_limit) & (head_element_ids[None, :] < head_dim), other=0.0)
        h_tile = tl.load(h_ptrs, mask=(head_element_ids[:, None] < head_dim) & (state_tile_element_ids[None, :] < state_dim), other=0.0)

        C_grad_contrib = tl.dot(y_grad_tile, h_tile)

        decay_cumsum_current = tl.load(decay_cumsum_ptrs, mask=chunk_tile_element_ids < chunk_size_limit, other=0.0)
        scale = tl.exp(decay_cumsum_current)

        C_grad_contrib *= scale[:, None]

        C_grad_acc += C_grad_contrib

        # Gradient for decay_cumsum from C path: sum(C_grad_contrib * C_tile, axis=1)
        decay_grad_from_C_contrib = tl.sum(C_grad_contrib * C_tile, axis=1) 
        tl.atomic_add(decay_grad_from_C_ptrs, decay_grad_from_C_contrib, mask=chunk_tile_element_ids < chunk_size_limit)

        y_grad_ptrs += y_grad_head_stride
        h_ptrs += h_head_stride
        decay_cumsum_ptrs += decay_cumsum_head_stride
        decay_grad_from_C_ptrs += decay_grad_from_C_head_stride

    # Store accumulated C_grad for this split
    C_grad_ptrs = C_grad_ptr + (chunk_tile_element_ids[:, None] * C_grad_seq_stride + state_tile_element_ids[None, :] * C_grad_state_element_stride)
    tl.store(C_grad_ptrs, C_grad_acc, mask=(chunk_tile_element_ids[:, None] < chunk_size_limit) & (state_tile_element_ids[None, :] < state_dim))


####################################################################################################


@triton.autotune(
    configs=[
        triton.Config({'CHUNK_TILE_Y_SIZE': 32, 'CHUNK_TILE_X_SIZE': 128}, num_stages=3, num_warps=4),
        triton.Config({'CHUNK_TILE_Y_SIZE': 128, 'CHUNK_TILE_X_SIZE': 32}, num_stages=3, num_warps=4),
        triton.Config({'CHUNK_TILE_Y_SIZE': 64, 'CHUNK_TILE_X_SIZE': 64}, num_stages=3, num_warps=4),
        triton.Config({'CHUNK_TILE_Y_SIZE': 64, 'CHUNK_TILE_X_SIZE': 32}, num_stages=3, num_warps=4),
        triton.Config({'CHUNK_TILE_Y_SIZE': 32, 'CHUNK_TILE_X_SIZE': 64}, num_stages=3, num_warps=4),
        triton.Config({'CHUNK_TILE_Y_SIZE': 32, 'CHUNK_TILE_X_SIZE': 32}, num_stages=3, num_warps=4),
    ],
    key=['chunk_size', 'head_dim'],
)
@triton.jit
def chunk_scan_backward_CB_kernel(
    u_ptr,  # (batch_size, seq_len, num_heads, head_dim)
    delta_ptr,  # (batch_size, num_heads, num_chunks, chunk_size)
    decay_cumsum_ptr,   # (batch_size, num_heads, num_chunks, chunk_size)
    y_grad_ptr, # (batch_size, seq_len, num_heads, head_dim)
    CB_grad_ptr,    # (batch_size, num_chunks, num_splits, num_groups, chunk_size, chunk_size)
    length_ptr, # (batch_size,)

    batch_size, seq_len, chunk_size, num_heads, head_dim, num_groups, num_heads_per_program,

    u_batch_stride, u_seq_stride, u_head_stride, u_head_element_stride,
    delta_batch_stride, delta_head_stride, delta_chunk_stride, delta_chunk_element_stride,
    decay_cumsum_batch_stride, decay_cumsum_head_stride, decay_cumsum_chunk_stride, decay_cumsum_chunk_element_stride,
    y_grad_batch_stride, y_grad_seq_stride, y_grad_head_stride, y_grad_head_element_stride,
    CB_grad_batch_stride, CB_grad_chunk_stride, CB_grad_split_stride, CB_grad_group_stride, CB_grad_chunk_y_stride, CB_grad_chunk_x_stride,
    length_batch_stride,

    CHUNK_TILE_Y_SIZE: tl.constexpr,
    CHUNK_TILE_X_SIZE: tl.constexpr,
    HEAD_DIM_ALIGNED: tl.constexpr
):
    # Map program IDs to batch, chunk, split/group, and 2D tile positions
    batch_chunk_id = tl.program_id(axis=1)
    chunk_id = batch_chunk_id // batch_size
    batch_id = batch_chunk_id - chunk_id * batch_size
    split_group_id = tl.program_id(axis=2)
    split_id = split_group_id // num_groups
    group_id = split_group_id - split_id * num_groups
    num_chunk_tile_x = tl.cdiv(chunk_size, CHUNK_TILE_X_SIZE)
    chunk_tile_y_id = tl.program_id(axis=0) // num_chunk_tile_x
    chunk_tile_x_id = tl.program_id(axis=0) % num_chunk_tile_x

    chunk_tile_y_element_ids = chunk_tile_y_id * CHUNK_TILE_Y_SIZE + tl.arange(0, CHUNK_TILE_Y_SIZE)
    chunk_tile_x_element_ids = chunk_tile_x_id * CHUNK_TILE_X_SIZE + tl.arange(0, CHUNK_TILE_X_SIZE)
    head_element_ids = tl.arange(0, HEAD_DIM_ALIGNED)

    # Advance pointers to batch, chunk, and head group
    head_start = group_id * (num_heads // num_groups) + split_id * num_heads_per_program

    u_ptr += batch_id * u_batch_stride + chunk_id * chunk_size * u_seq_stride + head_start * u_head_stride
    y_grad_ptr += batch_id * y_grad_batch_stride + chunk_id * chunk_size * y_grad_seq_stride + head_start * y_grad_head_stride
    delta_ptr += batch_id * delta_batch_stride + head_start * delta_head_stride + chunk_id * delta_chunk_stride
    decay_cumsum_ptr += batch_id * decay_cumsum_batch_stride + head_start * decay_cumsum_head_stride + chunk_id * decay_cumsum_chunk_stride
    CB_grad_ptr += batch_id * CB_grad_batch_stride + chunk_id * CB_grad_chunk_stride + group_id * CB_grad_group_stride + split_id * CB_grad_split_stride
    length_ptr += batch_id * length_batch_stride

    length = tl.load(length_ptr)
    chunk_size_limit = min(chunk_size, length - chunk_id * chunk_size)
    chunk_size_limit_x = min(chunk_size_limit, (chunk_tile_y_id + 1) * CHUNK_TILE_Y_SIZE)

    # If the tile is entirely above the diagonal (row < col), store zeros and return
    if chunk_tile_x_id * CHUNK_TILE_X_SIZE >= (chunk_tile_y_id + 1) * CHUNK_TILE_Y_SIZE:
        CB_grad_ptrs = CB_grad_ptr + (chunk_tile_y_element_ids[:, None] * CB_grad_chunk_y_stride + chunk_tile_x_element_ids[None, :] * CB_grad_chunk_x_stride)
        tl.store(CB_grad_ptrs, tl.zeros((CHUNK_TILE_Y_SIZE, CHUNK_TILE_X_SIZE), dtype=tl.float32),
                 mask=(chunk_tile_y_element_ids[:, None] < chunk_size) & (chunk_tile_x_element_ids[None, :] < chunk_size))
        return

    CB_grad_acc = tl.zeros((CHUNK_TILE_Y_SIZE, CHUNK_TILE_X_SIZE), dtype=tl.float32)

    # Pointers that advance across heads
    y_grad_ptrs = y_grad_ptr + (chunk_tile_y_element_ids[:, None] * y_grad_seq_stride + head_element_ids[None, :] * y_grad_head_element_stride)
    u_ptrs = u_ptr + (chunk_tile_x_element_ids[None, :] * u_seq_stride + head_element_ids[:, None] * u_head_element_stride)
    delta_ptrs = delta_ptr + chunk_tile_x_element_ids * delta_chunk_element_stride
    decay_cumsum_y_ptrs = decay_cumsum_ptr + chunk_tile_y_element_ids * decay_cumsum_chunk_element_stride
    decay_cumsum_x_ptrs = decay_cumsum_ptr + chunk_tile_x_element_ids * decay_cumsum_chunk_element_stride

    num_heads_iter = min(num_heads_per_program, num_heads // num_groups - split_id * num_heads_per_program)

    # Accumulate CB_grad = y_grad @ u, weighted by delta and decay scale
    for _ in range(num_heads_iter):
        y_grad_tile = tl.load(y_grad_ptrs, mask=(chunk_tile_y_element_ids[:, None] < chunk_size_limit) & (head_element_ids[None, :] < head_dim), other=0.0)
        u_tile = tl.load(u_ptrs, mask=(head_element_ids[:, None] < head_dim) & (chunk_tile_x_element_ids[None, :] < chunk_size_limit_x), other=0.0)

        # contrib = y_grad @ u (CHUNK_TILE_Y x CHUNK_TILE_X)
        contrib = tl.dot(y_grad_tile, u_tile)

        delta_x = tl.load(delta_ptrs, mask=chunk_tile_x_element_ids < chunk_size_limit_x, other=0.0)
        contrib *= delta_x[None, :]

        decay_cumsum_y = tl.load(decay_cumsum_y_ptrs, mask=chunk_tile_y_element_ids < chunk_size_limit, other=0.0)
        decay_cumsum_x = tl.load(decay_cumsum_x_ptrs, mask=chunk_tile_x_element_ids < chunk_size_limit_x, other=0.0)

        scale = tl.exp(tl.minimum(decay_cumsum_y[:, None] - decay_cumsum_x[None, :], 0.0))
        contrib *= scale

        CB_grad_acc += contrib

        y_grad_ptrs += y_grad_head_stride
        u_ptrs += u_head_stride
        delta_ptrs += delta_head_stride
        decay_cumsum_y_ptrs += decay_cumsum_head_stride
        decay_cumsum_x_ptrs += decay_cumsum_head_stride

    # Apply lower-triangular mask (row >= col)
    triangular_mask = chunk_tile_y_element_ids[:, None] >= chunk_tile_x_element_ids[None, :]
    CB_grad_acc = tl.where(triangular_mask, CB_grad_acc, 0.0)

    # Store accumulated CB_grad for this split
    CB_grad_ptrs = CB_grad_ptr + (chunk_tile_y_element_ids[:, None] * CB_grad_chunk_y_stride + chunk_tile_x_element_ids[None, :] * CB_grad_chunk_x_stride)
    tl.store(CB_grad_ptrs, CB_grad_acc, mask=(chunk_tile_y_element_ids[:, None] < chunk_size) & (chunk_tile_x_element_ids[None, :] < chunk_size))


####################################################################################################


@triton.autotune(
    configs=[
        triton.Config({'CHUNK_TILE_Y_SIZE': 128, 'STATE_TILE_SIZE': 256, 'CHUNK_TILE_X_SIZE': 64}, num_stages=3, num_warps=8),
        triton.Config({'CHUNK_TILE_Y_SIZE': 64, 'STATE_TILE_SIZE': 256, 'CHUNK_TILE_X_SIZE': 32}, num_stages=4, num_warps=4),
        triton.Config({'CHUNK_TILE_Y_SIZE': 128, 'STATE_TILE_SIZE': 128, 'CHUNK_TILE_X_SIZE': 32}, num_stages=4, num_warps=4),
        triton.Config({'CHUNK_TILE_Y_SIZE': 128, 'STATE_TILE_SIZE': 64, 'CHUNK_TILE_X_SIZE': 32}, num_stages=4, num_warps=4),
        triton.Config({'CHUNK_TILE_Y_SIZE': 64, 'STATE_TILE_SIZE': 128, 'CHUNK_TILE_X_SIZE': 32}, num_stages=4, num_warps=4),
        triton.Config({'CHUNK_TILE_Y_SIZE': 128, 'STATE_TILE_SIZE': 32, 'CHUNK_TILE_X_SIZE': 32}, num_stages=4, num_warps=4),
        triton.Config({'CHUNK_TILE_Y_SIZE': 64, 'STATE_TILE_SIZE': 32, 'CHUNK_TILE_X_SIZE': 32}, num_stages=5, num_warps=2),
        triton.Config({'CHUNK_TILE_Y_SIZE': 32, 'STATE_TILE_SIZE': 64, 'CHUNK_TILE_X_SIZE': 32}, num_stages=5, num_warps=2),
        triton.Config({'CHUNK_TILE_Y_SIZE': 64, 'STATE_TILE_SIZE': 64, 'CHUNK_TILE_X_SIZE': 32}, num_stages=4, num_warps=2),
    ],
    key=['chunk_size', 'state_dim'],
)
@triton.jit
def bmm_chunk_backward_kernel(
    C_ptr,  # (batch_size, seq_len, num_groups, state_dim)
    CB_grad_ptr,    # (batch_size, num_chunks, num_groups, chunk_size, chunk_size)
    B_grad_ptr, # (batch_size, seq_len, num_groups, state_dim)
    length_ptr, # (batch_size,)

    seq_len, chunk_size, state_dim, num_groups,

    C_batch_stride, C_seq_stride, C_group_stride, C_state_element_stride,
    CB_grad_batch_stride, CB_grad_chunk_stride, CB_group_stride, CB_chunk_y_element_stride, CB_chunk_x_element_stride,
    B_grad_batch_stride, B_grad_seq_stride, B_grad_group_stride, B_grad_state_element_stride,
    length_batch_stride,

    CHUNK_TILE_Y_SIZE: tl.constexpr,
    CHUNK_TILE_X_SIZE: tl.constexpr,
    STATE_TILE_SIZE: tl.constexpr,
):
    # Map program IDs to batch, chunk/group, and tile positions
    batch_id = tl.program_id(axis=1)
    chunk_group_id = tl.program_id(axis=2)
    chunk_id = chunk_group_id // num_groups
    group_id = chunk_group_id - chunk_id * num_groups
    num_state_tiles = tl.cdiv(state_dim, STATE_TILE_SIZE)
    chunk_tile_y_id = tl.program_id(axis=0) // num_state_tiles
    state_tile_id = tl.program_id(axis=0) % num_state_tiles

    chunk_tile_y_element_ids = chunk_tile_y_id * CHUNK_TILE_Y_SIZE + tl.arange(0, CHUNK_TILE_Y_SIZE)
    state_tile_element_ids = state_tile_id * STATE_TILE_SIZE + tl.arange(0, STATE_TILE_SIZE)
    chunk_tile_x_element_ids = tl.arange(0, CHUNK_TILE_X_SIZE)

    chunk_tile_y_element_ids = chunk_tile_y_id * CHUNK_TILE_Y_SIZE + tl.arange(0, CHUNK_TILE_Y_SIZE)
    state_tile_element_ids = state_tile_id * STATE_TILE_SIZE + tl.arange(0, STATE_TILE_SIZE)
    chunk_tile_x_element_ids = tl.arange(0, CHUNK_TILE_X_SIZE)

    # Advance pointers to batch, chunk, and group
    C_ptr += batch_id * C_batch_stride + chunk_id * chunk_size * C_seq_stride + group_id * C_group_stride
    CB_grad_ptr += batch_id * CB_grad_batch_stride + chunk_id * CB_grad_chunk_stride + group_id * CB_group_stride
    B_grad_ptr += batch_id * B_grad_batch_stride + chunk_id * chunk_size * B_grad_seq_stride + group_id * B_grad_group_stride
    length_ptr += batch_id * length_batch_stride

    length = tl.load(length_ptr)
    chunk_size_limit = min(chunk_size, length - chunk_id * chunk_size)
    acc = tl.zeros((CHUNK_TILE_Y_SIZE, STATE_TILE_SIZE), dtype=tl.float32)

    # Accumulate B_grad += CB_grad @ C (with swapped strides to match reference)
    CB_grad_ptrs = CB_grad_ptr + (chunk_tile_y_element_ids[:, None] * CB_chunk_x_element_stride + chunk_tile_x_element_ids[None, :] * CB_chunk_y_element_stride)
    C_ptrs = C_ptr + (chunk_tile_x_element_ids[:, None] * C_seq_stride + state_tile_element_ids[None, :] * C_state_element_stride)

    for reduce_idx in range(0, chunk_size_limit, CHUNK_TILE_X_SIZE):
        reduce_idx = tl.multiple_of(reduce_idx, CHUNK_TILE_X_SIZE)

        CB_grad_tile = tl.load(
            CB_grad_ptrs,
            mask=(chunk_tile_y_element_ids[:, None] < chunk_size_limit) & (chunk_tile_x_element_ids[None, :] < chunk_size_limit - reduce_idx), other=0.0
        )

        C_tile = tl.load(
            C_ptrs,
            mask=(chunk_tile_x_element_ids[:, None] < chunk_size_limit - reduce_idx) & (state_tile_element_ids[None, :] < state_dim), other=0.0
        )

        acc += tl.dot(CB_grad_tile, C_tile)

        CB_grad_ptrs += CHUNK_TILE_X_SIZE * CB_chunk_y_element_stride
        C_ptrs += CHUNK_TILE_X_SIZE * C_seq_stride

    # Add to existing B_grad (in-place update)
    B_grad_ptrs = B_grad_ptr + (chunk_tile_y_element_ids[:, None] * B_grad_seq_stride + state_tile_element_ids[None, :] * B_grad_state_element_stride)
    current = tl.load(
        B_grad_ptrs,
        mask=(chunk_tile_y_element_ids[:, None] < chunk_size_limit) & (state_tile_element_ids[None, :] < state_dim), other=0.0
    )
    updated = current + acc
    tl.store(
        B_grad_ptrs, updated,
        mask=(chunk_tile_y_element_ids[:, None] < chunk_size_limit) & (state_tile_element_ids[None, :] < state_dim)
    )


####################################################################################################


@triton.autotune(
    configs=[
        triton.Config({'CHUNK_TILE_Y_SIZE': 32, 'CHUNK_TILE_X_SIZE': 32}, num_stages=3, num_warps=4),
        triton.Config({'CHUNK_TILE_Y_SIZE': 64, 'CHUNK_TILE_X_SIZE': 32}, num_stages=3, num_warps=4),
        triton.Config({'CHUNK_TILE_Y_SIZE': 64, 'CHUNK_TILE_X_SIZE': 64}, num_stages=3, num_warps=4),
        triton.Config({'CHUNK_TILE_Y_SIZE': 128, 'CHUNK_TILE_X_SIZE': 32}, num_stages=3, num_warps=4),
        triton.Config({'CHUNK_TILE_Y_SIZE': 128, 'CHUNK_TILE_X_SIZE': 64}, num_stages=3, num_warps=4),
        triton.Config({'CHUNK_TILE_Y_SIZE': 128, 'CHUNK_TILE_X_SIZE': 128}, num_stages=3, num_warps=4),
    ],
    key=['chunk_size', 'head_dim'],
)
@triton.jit
def chunk_scan_backward_decay_cumsum_kernel(
    u_ptr,  # (batch_size, seq_len, num_heads, head_dim)
    delta_ptr,  # (batch_size, num_heads, num_chunks, chunk_size)
    decay_cumsum_ptr,   # (batch_size, num_heads, num_chunks, chunk_size)
    y_grad_ptr, # (batch_size, seq_len, num_heads, head_dim)
    CB_ptr, # (batch_size, num_chunks, num_groups, chunk_size, chunk_size)
    decay_grad_ptr,  # (batch_size, num_heads, num_chunks, num_chunk_tiles, chunk_size)
    length_ptr,
    
    batch_size, seq_len, chunk_size, head_dim, num_heads_per_group,

    u_batch_stride, u_seq_stride, u_head_stride, u_head_element_stride,
    delta_batch_stride, delta_head_stride, delta_chunk_stride, delta_chunk_element_stride,
    decay_cumsum_batch_stride, decay_cumsum_head_stride, decay_cumsum_chunk_stride, decay_cumsum_chunk_element_stride,
    y_grad_batch_stride, y_grad_seq_stride, y_grad_head_stride, y_grad_head_element_stride,
    CB_batch_stride, CB_chunk_stride, CB_group_stride, CB_chunk_y_element_stride, CB_chunk_x_element_stride,
    decay_grad_batch_stride, decay_grad_head_stride, decay_grad_chunk_stride, decay_grad_chunk_tile_stride, decay_grad_chunk_element_stride,
    length_batch_stride,

    CHUNK_TILE_Y_SIZE: tl.constexpr,
    CHUNK_TILE_X_SIZE: tl.constexpr,
    HEAD_DIM_ALIGNED: tl.constexpr
):
    # Map program IDs to batch, chunk, head, and row tile
    batch_chunk_id = tl.program_id(axis=1)
    chunk_id = batch_chunk_id // batch_size
    batch_id = batch_chunk_id - chunk_id * batch_size
    head_id = tl.program_id(axis=2)
    chunk_tile_y_id = tl.program_id(axis=0)

    chunk_tile_y_element_ids = chunk_tile_y_id * CHUNK_TILE_Y_SIZE + tl.arange(0, CHUNK_TILE_Y_SIZE)
    chunk_tile_x_element_ids = tl.arange(0, CHUNK_TILE_X_SIZE)
    head_element_ids = tl.arange(0, HEAD_DIM_ALIGNED)

    # Advance pointers to batch, chunk, and head
    u_ptr += batch_id * u_batch_stride + chunk_id * chunk_size * u_seq_stride + head_id * u_head_stride
    y_grad_ptr += batch_id * y_grad_batch_stride + chunk_id * chunk_size * y_grad_seq_stride + head_id * y_grad_head_stride
    delta_ptr += batch_id * delta_batch_stride + head_id * delta_head_stride + chunk_id * delta_chunk_stride
    decay_cumsum_ptr += batch_id * decay_cumsum_batch_stride + head_id * decay_cumsum_head_stride + chunk_id * decay_cumsum_chunk_stride
    group_id = head_id // num_heads_per_group
    CB_ptr += batch_id * CB_batch_stride + chunk_id * CB_chunk_stride + group_id * CB_group_stride
    decay_grad_ptr += (batch_id * decay_grad_batch_stride + head_id * decay_grad_head_stride + chunk_id * decay_grad_chunk_stride + chunk_tile_y_id * decay_grad_chunk_tile_stride)
    length_ptr += batch_id * length_batch_stride

    length = tl.load(length_ptr)
    chunk_size_limit = min(chunk_size, length - chunk_id * chunk_size)

    # Load fixed row-tile of y_grad and decay_cumsum (row)
    y_grad_ptrs = y_grad_ptr + (chunk_tile_y_element_ids[:, None] * y_grad_seq_stride + head_element_ids[None, :] * y_grad_head_element_stride)
    decay_cumsum_ptrs = decay_cumsum_ptr + chunk_tile_y_element_ids * decay_cumsum_chunk_element_stride

    y_grad_tile = tl.load(
        y_grad_ptrs,
        mask=(chunk_tile_y_element_ids[:, None] < chunk_size_limit) & (head_element_ids[None, :] < head_dim),
        other=0.0
    )
    decay_cumsum_y = tl.load(decay_cumsum_ptrs, mask=chunk_tile_y_element_ids < chunk_size_limit, other=0.0)

    ## Initialize running row sum and zero first position of the tile
    rowsum = tl.zeros((CHUNK_TILE_Y_SIZE,), dtype=tl.float32)
    tl.store(decay_grad_ptr, 0.0) 

    # Column-moving pointers
    u_ptrs = u_ptr + (chunk_tile_x_element_ids[None, :] * u_seq_stride + head_element_ids[:, None] * u_head_element_stride)
    delta_ptrs = delta_ptr + chunk_tile_x_element_ids * delta_chunk_element_stride
    CB_ptrs = CB_ptr + (chunk_tile_y_element_ids[:, None] * CB_chunk_y_element_stride + chunk_tile_x_element_ids[None, :] * CB_chunk_x_element_stride)
    decay_grad_ptrs = (decay_grad_ptr + chunk_tile_x_element_ids * decay_grad_chunk_element_stride)

    lo = 0
    hi = (chunk_tile_y_id + 1) * CHUNK_TILE_Y_SIZE

    # Accumulate decay gradient while maintaining rowsum
    for start_c in range(lo, hi, CHUNK_TILE_X_SIZE):
        start_c = tl.multiple_of(start_c, CHUNK_TILE_X_SIZE)

        u_tile = tl.load(
            u_ptrs,
            mask=(head_element_ids[:, None] < head_dim) & (chunk_tile_x_element_ids[None, :] < chunk_size_limit - start_c),
            other=0.0
        )

        # contrib = y_grad @ u (CHUNK_TILE_Y x CHUNK_TILE_X)
        contrib = tl.dot(y_grad_tile, u_tile)

        delta_x = tl.load(delta_ptrs, mask=chunk_tile_x_element_ids < chunk_size - start_c, other=0.0)
        contrib *= delta_x[None, :]

        CB_tile = tl.load(
            CB_ptrs,
            mask=(chunk_tile_y_element_ids[:, None] < chunk_size_limit) & (chunk_tile_x_element_ids[None, :] < chunk_size - start_c),
            other=0.0
        )
        contrib *= CB_tile

        decay_cumsum_x = tl.load(
            decay_cumsum_ptr + (start_c + chunk_tile_x_element_ids) * decay_cumsum_chunk_element_stride,
            mask=chunk_tile_x_element_ids < chunk_size - start_c, other=0.0
        )
        scale = tl.exp(tl.minimum(decay_cumsum_y[:, None] - decay_cumsum_x[None, :], 0.0))
        contrib *= scale

        # Mask: only keep i >= j + 1 (strictly lower triangular with shifted diagonal)
        tri_mask = chunk_tile_y_element_ids[:, None] >= (start_c + chunk_tile_x_element_ids[None, :] + 1)
        contrib = tl.where(tri_mask, contrib, 0.0)

        rowsum_new = rowsum + tl.sum(contrib, axis=1)
        contrib = rowsum[:, None] + tl.cumsum(contrib, axis=1)
        rowsum = rowsum_new
        contrib = tl.where(tri_mask, contrib, 0.0)
        decay_grad_col = tl.sum(contrib, axis=0)

        # Store with offset +1 (exclusive cumsum)
        tl.store(
            decay_grad_ptrs + decay_grad_chunk_element_stride, decay_grad_col,
            mask=chunk_tile_x_element_ids < chunk_size - start_c - 1
        )

        u_ptrs += CHUNK_TILE_X_SIZE * u_seq_stride
        delta_ptrs += CHUNK_TILE_X_SIZE * delta_chunk_element_stride
        CB_ptrs += CHUNK_TILE_X_SIZE * CB_chunk_x_element_stride
        decay_grad_ptrs += CHUNK_TILE_X_SIZE * decay_grad_chunk_element_stride

    # Zero out remaining columns beyond the current row tile
    for start_c in range(hi, chunk_size, CHUNK_TILE_X_SIZE):
        tl.store(
            decay_grad_ptrs + decay_grad_chunk_element_stride, tl.zeros((CHUNK_TILE_X_SIZE,), dtype=tl.float32),
            mask=chunk_tile_x_element_ids < chunk_size - start_c - 1
        )
        decay_grad_ptrs += CHUNK_TILE_X_SIZE * decay_grad_chunk_element_stride


####################################################################################################


@triton.autotune(
    configs=[
        triton.Config({'HEAD_GROUP_SIZE': 1}, pre_hook=reset_buffers(["A_grad_ptr", "delta_bias_grad_ptr"])),
        triton.Config({'HEAD_GROUP_SIZE': 2}, pre_hook=reset_buffers(["A_grad_ptr", "delta_bias_grad_ptr"])),
        triton.Config({'HEAD_GROUP_SIZE': 4}, pre_hook=reset_buffers(["A_grad_ptr", "delta_bias_grad_ptr"])),
        triton.Config({'HEAD_GROUP_SIZE': 8}, pre_hook=reset_buffers(["A_grad_ptr", "delta_bias_grad_ptr"])),
        triton.Config({'HEAD_GROUP_SIZE': 16}, pre_hook=reset_buffers(["A_grad_ptr", "delta_bias_grad_ptr"])),
        triton.Config({'HEAD_GROUP_SIZE': 32}, pre_hook=reset_buffers(["A_grad_ptr", "delta_bias_grad_ptr"])),
        triton.Config({'HEAD_GROUP_SIZE': 64}, pre_hook=reset_buffers(["A_grad_ptr", "delta_bias_grad_ptr"])),
    ],
    key=['chunk_size', 'num_heads'],
)
@triton.jit
def chunk_cumsum_backward_kernel(
    A_ptr,  # (num_heads,)
    delta_raw_ptr,  # (batch_size, seq_len, num_heads)
    delta_bias_ptr, # (num_heads)
    delta_grad_ptr, # (batch_size, num_heads, num_chunks, chunk_size)
    decay_grad_total_ptr,  # (batch_size, num_heads, num_chunks, chunk_size)
    delta_bias_grad_ptr,    # (num_heads,)
    delta_raw_grad_ptr, # (batch_size, seq_len, num_heads)
    A_grad_ptr, # (num_heads,)
    length_ptr, # (batch_size,)

    seq_len, num_heads, chunk_size, delta_min, delta_max,

    A_head_stride,
    delta_raw_batch_stride, delta_raw_seq_stride, delta_raw_head_stride,
    delta_bias_head_stride,
    delta_grad_batch_stride, delta_grad_head_stride, delta_grad_chunk_stride, delta_grad_chunk_element_stride,
    decay_grad_total_batch_stride, decay_grad_total_head_stride, decay_grad_total_chunk_stride, decay_grad_total_chunk_element_stride,
    delta_bias_grad_head_stride,
    delta_raw_grad_batch_stride, delta_raw_grad_seq_stride, delta_raw_grad_head_stride,
    A_grad_head_stride,
    length_batch_stride,

    USE_DELTA_SOFTPLUS: tl.constexpr,
    HAS_DELTA_BIAS: tl.constexpr,
    CHUNK_SIZE_ALIGNED: tl.constexpr,
    HEAD_GROUP_SIZE: tl.constexpr
):
    # Map program IDs to batch, chunk, and head group
    batch_id = tl.program_id(axis=0)
    chunk_id = tl.program_id(axis=1)
    head_group_id = tl.program_id(axis=2)

    head_ids = head_group_id * HEAD_GROUP_SIZE + tl.arange(0, HEAD_GROUP_SIZE)
    chunk_element_ids = tl.arange(0, CHUNK_SIZE_ALIGNED)

    # Advance pointers to batch and chunk
    delta_grad_ptr += batch_id * delta_grad_batch_stride + chunk_id * delta_grad_chunk_stride
    decay_grad_total_ptr += batch_id * decay_grad_total_batch_stride + chunk_id * decay_grad_total_chunk_stride
    delta_raw_ptr += batch_id * delta_raw_batch_stride + chunk_id * chunk_size * delta_raw_seq_stride
    delta_raw_grad_ptr += batch_id * delta_raw_grad_batch_stride + chunk_id * chunk_size * delta_raw_grad_seq_stride
    length_ptr += batch_id * length_batch_stride

    length = tl.load(length_ptr)
    chunk_size_limit = min(chunk_size, length - chunk_id * chunk_size)

    head_mask = head_ids < num_heads

    # Load A (same for every chunk element)
    A_ptrs = A_ptr + head_ids * A_head_stride
    A_grad_ptrs = A_grad_ptr + head_ids * A_grad_head_stride
    A = tl.load(A_ptrs, mask=head_mask, other=0.0).to(tl.float32)

    # Pointers per head and chunk position
    delta_grad_ptrs = (delta_grad_ptr + head_ids[:, None] * delta_grad_head_stride + chunk_element_ids[None, :] * delta_grad_chunk_element_stride)
    decay_grad_total_ptrs = (decay_grad_total_ptr + head_ids[:, None] * decay_grad_total_head_stride + chunk_element_ids[None, :] * decay_grad_total_chunk_element_stride)
    delta_raw_ptrs = (delta_raw_ptr + head_ids[:, None] * delta_raw_head_stride + chunk_element_ids[None, :] * delta_raw_seq_stride)
    delta_raw_grad_ptrs = (delta_raw_grad_ptr + head_ids[:, None] * delta_raw_grad_head_stride + chunk_element_ids[None, :] * delta_raw_grad_seq_stride)

    # Load chunk data
    delta_grad = tl.load(
        delta_grad_ptrs,
        mask=head_mask[:, None] & (chunk_element_ids[None, :] < chunk_size_limit), other=0.0
    )
    decay_grad_total = tl.load(
        decay_grad_total_ptrs,
        mask=head_mask[:, None] & (chunk_element_ids[None, :] < chunk_size_limit), other=0.0
    )
    delta_raw = tl.load(
        delta_raw_ptrs,
        mask=head_mask[:, None] & (chunk_element_ids[None, :] < chunk_size_limit), other=0.0
    )

    # Compute gradient for delta_raw: decay_grad_total * A + delta_grad
    delta_raw_grad = decay_grad_total * A[:, None] + delta_grad

    # Build final delta (with bias and softplus)
    delta = delta_raw
    if HAS_DELTA_BIAS:
        delta_bias_ptrs = delta_bias_ptr + head_ids * delta_bias_head_stride
        delta_bias = tl.load(delta_bias_ptrs, mask=head_mask, other=0.0).to(tl.float32)
        delta += delta_bias[:, None]

    if USE_DELTA_SOFTPLUS:
        delta_presoftplus = delta
        delta = tl.where(delta_presoftplus <= 20.0, softplus(delta_presoftplus), delta_presoftplus)

    # Clamp and compute mask for positions that were clamped
    clamp_mask = (delta < delta_min) | (delta > delta_max)
    delta = tl.minimum(tl.maximum(delta, delta_min), delta_max)

    # Zero gradient for clamped positions
    delta_raw_grad = tl.where(clamp_mask & head_mask[:, None] & (chunk_element_ids[None, :] < chunk_size_limit), 0.0, delta_raw_grad)

    # Multiply by softplus derivative if needed
    if USE_DELTA_SOFTPLUS:
        delta_raw_grad = tl.where(delta_presoftplus <= 20.0, delta_raw_grad * tl.sigmoid(delta_presoftplus), delta_raw_grad)

    # Store gradient for delta_raw
    tl.store(delta_raw_grad_ptrs, delta_raw_grad, mask=head_mask[:, None] & (chunk_element_ids[None, :] < chunk_size_limit))

    # Gradient for A: sum(decay_grad_total * delta, axis=1)
    A_grad_contrib = tl.sum(decay_grad_total * delta, axis=1)
    tl.atomic_add(A_grad_ptrs, A_grad_contrib, mask=head_mask)

    # Gradient for delta bias
    if HAS_DELTA_BIAS:
        delta_bias_grad_contrib = tl.sum(delta_raw_grad, axis=1)
        delta_bias_grad_ptrs = delta_bias_grad_ptr + head_ids * delta_bias_grad_head_stride
        tl.atomic_add(delta_bias_grad_ptrs, delta_bias_grad_contrib, mask=head_mask)
