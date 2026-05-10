import triton
import triton.language as tl

from selective_attention.triton_kernels.softplus import softplus


@triton.autotune(
    configs=[
        triton.Config({'HEAD_GROUP_SIZE': 1}),
        triton.Config({'HEAD_GROUP_SIZE': 2}),
        triton.Config({'HEAD_GROUP_SIZE': 4}),
        triton.Config({'HEAD_GROUP_SIZE': 8}),
        triton.Config({'HEAD_GROUP_SIZE': 16}),
        triton.Config({'HEAD_GROUP_SIZE': 32}),
        triton.Config({'HEAD_GROUP_SIZE': 64}),
    ],
    key=["chunk_size", "num_heads"]
)
@triton.jit
def chunk_cumsum_forward_kernel(
    A_ptr,  # (num_heads,)
    delta_raw_ptr,  # (batch_size, seq_len, num_heads)
    delta_bias_ptr, # (num_heads,)
    delta_ptr,  # (batch_size, num_heads, num_chunks, chunk_size)
    decay_cumsum_ptr,   # (batch_size, num_heads, num_chunks, chunk_size)
    length_ptr, # (batch_size)
    
    seq_len, chunk_size, num_heads, delta_min, delta_max,
    
    A_head_stride,
    delta_raw_batch_stride, delta_raw_seq_stride, delta_raw_head_stride,
    delta_bias_head_stride,
    delta_batch_stride, delta_head_stride, delta_chunk_stride, delta_chunk_element_stride,
    decay_cumsum_batch_stride, decay_cumsum_head_stride, decay_cumsum_chunk_stride, decay_cumsum_chunk_element_stride,
    length_batch_stride,

    USE_DELTA_SOFTPLUS: tl.constexpr,
    HAS_DELTA_BIAS: tl.constexpr,
    HEAD_GROUP_SIZE: tl.constexpr,
    CHUNK_SIZE_ALIGNED: tl.constexpr
):
    # Map program IDs to batch, chunk, and head group
    batch_id = tl.program_id(axis=0)
    chunk_id = tl.program_id(axis=1)
    head_group_id = tl.program_id(axis=2)
    
    head_ids = head_group_id * HEAD_GROUP_SIZE + tl.arange(0, HEAD_GROUP_SIZE)
    chunk_element_ids = tl.arange(0, CHUNK_SIZE_ALIGNED)
    
    # Advance pointers to the current batch and chunk
    delta_raw_ptr += batch_id * delta_raw_batch_stride + chunk_id * chunk_size * delta_raw_seq_stride
    delta_ptr += batch_id * delta_batch_stride + chunk_id * delta_chunk_stride
    decay_cumsum_ptr += batch_id * decay_cumsum_batch_stride + chunk_id * decay_cumsum_chunk_stride
    length_ptr += batch_id * length_batch_stride

    A_ptrs = A_ptr + (head_ids * A_head_stride)
    delta_raw_ptrs = delta_raw_ptr + (head_ids[:, None] * delta_raw_head_stride + chunk_element_ids[None, :] * delta_raw_seq_stride)
    delta_ptrs = delta_ptr + (head_ids[:, None] * delta_head_stride + chunk_element_ids[None, :] * delta_chunk_element_stride) 
    decay_cumsum_ptrs = decay_cumsum_ptr + (head_ids[:, None] * decay_cumsum_head_stride + chunk_element_ids[None, :] * decay_cumsum_chunk_element_stride)

    # Load parameters and raw delta
    length = tl.load(length_ptr)
    chunk_size_limit = min(chunk_size, length - chunk_id * chunk_size)
    A = tl.load(A_ptrs, mask=head_ids < num_heads, other=0.0)
    delta_raw = tl.load(delta_raw_ptrs, mask=(head_ids[:, None] < num_heads) & (chunk_element_ids[None, :] < chunk_size_limit), other=0.0)

    # delta = delta_raw + delta_bias (if present)
    if HAS_DELTA_BIAS:
        delta_bias_ptrs = delta_bias_ptr + head_ids * delta_bias_head_stride
        delta_bias = tl.load(delta_bias_ptrs, mask=head_ids < num_heads, other=0.0)
        delta = delta_raw + delta_bias[:, None]
    else:
        delta = delta_raw

    # Apply softplus and clamp to build the final delta
    if USE_DELTA_SOFTPLUS:
        delta = tl.where(delta <= 20.0, softplus(delta), delta)
    
    delta = tl.minimum(tl.maximum(delta, delta_min), delta_max)
    delta = tl.where((head_ids[:, None] < num_heads) & (chunk_element_ids[None, :] < chunk_size_limit), delta, 0.0)
    
    # Compute decay_cumsum = cumsum(delta * A)
    decay = delta * A[:, None]
    decay_cumsum = tl.cumsum(decay, axis=1)
    
    tl.store(decay_cumsum_ptrs, decay_cumsum, mask=(head_ids[:, None] < num_heads) & (chunk_element_ids[None, :] < chunk_size))
    tl.store(delta_ptrs, delta, mask=(head_ids[:, None] < num_heads) & (chunk_element_ids[None, :] < chunk_size))


####################################################################################################


@triton.autotune(
    configs=[
        triton.Config({'HEAD_TILE_SIZE': 128, 'STATE_TILE_SIZE': 256, 'CHUNK_TILE_SIZE': 64}, num_stages=3, num_warps=8),
        triton.Config({'HEAD_TILE_SIZE': 64, 'STATE_TILE_SIZE': 256, 'CHUNK_TILE_SIZE': 32}, num_stages=4, num_warps=4),
        triton.Config({'HEAD_TILE_SIZE': 128, 'STATE_TILE_SIZE': 128, 'CHUNK_TILE_SIZE': 32}, num_stages=4, num_warps=4),
        triton.Config({'HEAD_TILE_SIZE': 128, 'STATE_TILE_SIZE': 64, 'CHUNK_TILE_SIZE': 32}, num_stages=4, num_warps=4),
        triton.Config({'HEAD_TILE_SIZE': 64, 'STATE_TILE_SIZE': 128, 'CHUNK_TILE_SIZE': 32}, num_stages=4, num_warps=4),
        triton.Config({'HEAD_TILE_SIZE': 128, 'STATE_TILE_SIZE': 32, 'CHUNK_TILE_SIZE': 32}, num_stages=4, num_warps=4),
        triton.Config({'HEAD_TILE_SIZE': 64, 'STATE_TILE_SIZE': 32, 'CHUNK_TILE_SIZE': 32}, num_stages=5, num_warps=2),
        triton.Config({'HEAD_TILE_SIZE': 32, 'STATE_TILE_SIZE': 64, 'CHUNK_TILE_SIZE': 32}, num_stages=5, num_warps=2),
        triton.Config({'HEAD_TILE_SIZE': 64, 'STATE_TILE_SIZE': 64, 'CHUNK_TILE_SIZE': 32}, num_stages=4, num_warps=2),
    ],
    key=['head_dim', 'state_dim', 'chunk_size'],
)
@triton.jit
def chunk_state_forward_kernel(
    u_ptr,  # (batch_size, seq_len, num_heads, head_dim) 
    B_ptr,  # (batch_size, seq_len, num_groups, state_dim)
    h_ptr,  # (batch_size, num_chunks, num_heads, head_dim, state_dim)
    delta_ptr,  # (batch_size, num_heads, num_chunks, chunk_size)
    decay_cumsum_ptr, # (batch_size, num_heads, num_chunks, chunk_size)

    batch_size, seq_len, chunk_size, head_dim, state_dim, num_heads_per_group,

    u_batch_stride, u_seq_stride, u_head_stride, u_head_element_stride,
    B_batch_stride, B_seq_stride, B_group_stride, B_state_stride,
    h_batch_stride, h_chunk_stride, h_head_stride, h_head_element_stride, h_state_stride,
    delta_batch_stride, delta_head_stride, delta_chunk_stride, delta_chunk_element_stride,
    decay_cumsum_batch_stride, decay_cumsum_head_stride, decay_cumsum_chunk_stride, decay_cumsum_chunk_element_stride,

    HEAD_TILE_SIZE: tl.constexpr,
    STATE_TILE_SIZE: tl.constexpr,
    CHUNK_TILE_SIZE: tl.constexpr
):
    # Map program IDs to batch, chunk, head, and tile positions
    batch_chunk_id = tl.program_id(axis=1)
    chunk_id = batch_chunk_id // batch_size
    batch_id = batch_chunk_id - chunk_id * batch_size
    head_id = tl.program_id(axis=2)
    num_state_tiles = tl.cdiv(state_dim, STATE_TILE_SIZE)
    head_tile_id = tl.program_id(axis=0) // num_state_tiles
    state_tile_id = tl.program_id(axis=0) % num_state_tiles
    chunk_size_limit = min(chunk_size, seq_len - chunk_id * chunk_size)

    head_element_ids = head_tile_id * HEAD_TILE_SIZE + tl.arange(0, HEAD_TILE_SIZE)
    state_element_ids = state_tile_id * STATE_TILE_SIZE + tl.arange(0, STATE_TILE_SIZE)
    chunk_tile_element_ids = tl.arange(0, CHUNK_TILE_SIZE)

    # Advance pointers to the current batch, chunk, head, and group
    u_ptr += batch_id * u_batch_stride + chunk_id * chunk_size * u_seq_stride + head_id * u_head_stride
    B_ptr += batch_id * B_batch_stride + chunk_id * chunk_size * B_seq_stride + (head_id // num_heads_per_group) * B_group_stride
    h_ptr += batch_id * h_batch_stride + chunk_id * h_chunk_stride + head_id * h_head_stride
    delta_ptr += batch_id * delta_batch_stride + chunk_id * delta_chunk_stride + head_id * delta_head_stride
    decay_cumsum_ptr += batch_id * decay_cumsum_batch_stride + head_id * decay_cumsum_head_stride + chunk_id * decay_cumsum_chunk_stride

    u_ptrs = u_ptr + (head_element_ids[:, None] * u_head_element_stride + chunk_tile_element_ids[None, :] * u_seq_stride)
    B_ptrs = B_ptr + (state_element_ids[None, :] * B_state_stride + chunk_tile_element_ids[:, None] * B_seq_stride)
    h_ptrs = h_ptr + (head_element_ids[:, None] * h_head_element_stride + state_element_ids[None, :] * h_state_stride)
    delta_ptrs = delta_ptr + (chunk_tile_element_ids * delta_chunk_element_stride)
    decay_cumsum_ptrs = decay_cumsum_ptr + chunk_tile_element_ids * decay_cumsum_chunk_element_stride
    
    # Pre-load the last decay cumsum for scaling inside the chunk
    decay_cumsum_last = tl.load(decay_cumsum_ptr + (chunk_size - 1) * decay_cumsum_chunk_element_stride)
    
    # Accumulate h = sum over chunk positions: u @ B^T, weighted by delta and decay
    h = tl.zeros((HEAD_TILE_SIZE, STATE_TILE_SIZE), dtype=tl.float32)
    for chunk_tile_id in range((chunk_size_limit - 1) // CHUNK_TILE_SIZE + 1):
        chunk_tile_size_limit = chunk_size_limit - chunk_tile_id * CHUNK_TILE_SIZE
        u = tl.load(
            u_ptrs + chunk_tile_id * CHUNK_TILE_SIZE * u_seq_stride,
            mask=(head_element_ids[:, None] < head_dim) & (chunk_tile_element_ids[None, :] < chunk_tile_size_limit), other=0.0
        )
        B = tl.load(
            B_ptrs + chunk_tile_id * CHUNK_TILE_SIZE * B_seq_stride,
            mask=(chunk_tile_element_ids[:, None] < chunk_tile_size_limit) & (state_element_ids[None, :] < state_dim), other=0.0
        )
        decay_cumsum = tl.load(
            decay_cumsum_ptrs + chunk_tile_id * CHUNK_TILE_SIZE * decay_cumsum_chunk_element_stride,
            mask=chunk_tile_element_ids < chunk_tile_size_limit, other=0.0
        )
        delta = tl.load(
            delta_ptrs + chunk_tile_id * CHUNK_TILE_SIZE * delta_chunk_element_stride,
            mask=chunk_tile_element_ids < chunk_tile_size_limit, other=0.0
        )
        # Weight: exp(decay_cumsum_last - decay_cumsum) * delta
        scale = tl.exp(tl.minimum((decay_cumsum_last - decay_cumsum), 0.0)) * delta
        B *= scale[:, None]
        h += tl.dot(u, B)
    
    tl.store(
        h_ptrs, h,
        mask=(head_element_ids[:, None] < head_dim) & (state_element_ids[None, :] < state_dim)
    )


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
def state_passing_forward_kernel(
    h_ptr,  # (batch_size, num_chunks, num_heads, head_dim * state_dim)
    h_init_ptr, # (batch_size, num_heads, head_dim * state_dim)
    h_last_ptr, # (batch_size, num_heads, head_dim * state_dim)
    decay_last_ptr, # (batch_size, num_heads, num_chunks)

    num_chunks, head_state_dim,

    h_batch_stride, h_chunk_stride, h_head_stride, h_head_state_element_stride,
    h_init_batch_stride, h_init_head_stride, h_init_head_state_element_stride,
    h_last_batch_stride, h_last_head_stride, h_last_head_state_element_stride,
    decay_last_batch_stride, decay_last_head_stride, decay_last_chunk_stride,

    HAS_H_INIT: tl.constexpr,
    HEAD_STATE_GROUP_SIZE: tl.constexpr,
):
    # Map program IDs to batch, head, and head-state group
    batch_id = tl.program_id(axis=0)
    head_id = tl.program_id(axis=1)
    head_state_group_id = tl.program_id(axis=2)

    head_state_element_ids = head_state_group_id * HEAD_STATE_GROUP_SIZE + tl.arange(0, HEAD_STATE_GROUP_SIZE)

    # Advance pointers to the current batch and head
    h_ptr += batch_id * h_batch_stride + head_id * h_head_stride
    h_last_ptr += batch_id * h_last_batch_stride + head_id * h_last_head_stride
    decay_last_ptr += batch_id * decay_last_batch_stride + head_id * decay_last_head_stride
    if HAS_H_INIT:
        h_init_ptr += batch_id * h_init_batch_stride + head_id * h_init_head_stride

    h_ptrs = h_ptr + head_state_element_ids * h_head_state_element_stride
    h_last_ptrs = h_last_ptr + head_state_element_ids * h_last_head_state_element_stride
    
    # Initialize previous state from h_init or zeros
    if HAS_H_INIT:
        h_init_ptrs = h_init_ptr + head_state_element_ids * h_init_head_state_element_stride
        prev_chunk_h = tl.load(h_init_ptrs, mask=head_state_element_ids < head_state_dim, other=0.0)
    else:
        prev_chunk_h = tl.zeros((HEAD_STATE_GROUP_SIZE,), dtype=tl.float32)
    
    # Propagate state across chunks: h_next = scale * h_prev + h_current
    for chunk_id in range(num_chunks):
        chunk_h = tl.load(h_ptrs + chunk_id * h_chunk_stride, mask=head_state_element_ids < head_state_dim, other=0.0)
        tl.store(h_ptrs + chunk_id * h_chunk_stride, prev_chunk_h, mask=head_state_element_ids < head_state_dim)
        decay_last = tl.load(decay_last_ptr + chunk_id * decay_last_chunk_stride)
        scale = tl.exp(decay_last)
        prev_chunk_h = scale * prev_chunk_h + chunk_h
    tl.store(h_last_ptrs, prev_chunk_h, mask=head_state_element_ids < head_state_dim)


####################################################################################################


@triton.autotune(
    configs=[
        triton.Config({'CHUNK_TILE_Y_SIZE': 128, 'CHUNK_TILE_X_SIZE': 256, 'STATE_TILE_SIZE': 64}, num_stages=3, num_warps=8),
        triton.Config({'CHUNK_TILE_Y_SIZE': 64, 'CHUNK_TILE_X_SIZE': 256, 'STATE_TILE_SIZE': 32}, num_stages=4, num_warps=4),
        triton.Config({'CHUNK_TILE_Y_SIZE': 128, 'CHUNK_TILE_X_SIZE': 128, 'STATE_TILE_SIZE': 32}, num_stages=4, num_warps=4),
        triton.Config({'CHUNK_TILE_Y_SIZE': 128, 'CHUNK_TILE_X_SIZE': 64, 'STATE_TILE_SIZE': 32}, num_stages=4, num_warps=4),
        triton.Config({'CHUNK_TILE_Y_SIZE': 64, 'CHUNK_TILE_X_SIZE': 128, 'STATE_TILE_SIZE': 32}, num_stages=4, num_warps=4),
        triton.Config({'CHUNK_TILE_Y_SIZE': 128, 'CHUNK_TILE_X_SIZE': 32, 'STATE_TILE_SIZE': 32}, num_stages=4, num_warps=4),
        triton.Config({'CHUNK_TILE_Y_SIZE': 64, 'CHUNK_TILE_X_SIZE': 32, 'STATE_TILE_SIZE': 32}, num_stages=5, num_warps=2),
        triton.Config({'CHUNK_TILE_Y_SIZE': 32, 'CHUNK_TILE_X_SIZE': 64, 'STATE_TILE_SIZE': 32}, num_stages=5, num_warps=2),
        triton.Config({'CHUNK_TILE_Y_SIZE': 64, 'CHUNK_TILE_X_SIZE': 64, 'STATE_TILE_SIZE': 32}, num_stages=4, num_warps=2),
    ],
    key=['chunk_size', 'STATE_TILE_SIZE', 'IS_CAUSAL'],
)
@triton.jit
def bmm_chunk_forward_kernel(
    B_ptr,  # (batch_size, seq_len, num_groups, state_dim)
    C_ptr,  # (batch_size, seq_len, num_groups, state_dim)
    CB_ptr, # (batch_size, num_chunks, num_groups, chunk_size, chunk_size)

    seq_len, chunk_size, num_groups, state_dim,

    B_batch_stride, B_seq_stride, B_group_stride, B_state_element_stride,
    C_batch_stride, C_seq_stride, C_group_stride, C_state_element_stride,
    CB_batch_stride, CB_chunk_stride, CB_group_stride, CB_chunk_y_element_stride, CB_chunk_x_element_stride,

    IS_CAUSAL: tl.constexpr,
    CHUNK_TILE_X_SIZE: tl.constexpr,
    CHUNK_TILE_Y_SIZE: tl.constexpr,
    STATE_TILE_SIZE: tl.constexpr
):
    # Map program IDs to batch, chunk/group, and 2D tile positions
    batch_id = tl.program_id(axis=1)
    chunk_group_id = tl.program_id(axis=2)
    chunk_id = chunk_group_id // num_groups
    group_id = chunk_group_id - chunk_id * num_groups
    num_chunk_tile_x = tl.cdiv(chunk_size, CHUNK_TILE_X_SIZE)
    chunk_tile_y_id = tl.program_id(axis=0) // num_chunk_tile_x
    chunk_tile_x_id =  tl.program_id(axis=0) % num_chunk_tile_x
    chunk_size_limit = min(chunk_size, seq_len - chunk_id * chunk_size)
    
    # If causal and the tile is entirely above the diagonal, skip
    if IS_CAUSAL:
        if chunk_tile_x_id * CHUNK_TILE_X_SIZE >= (chunk_tile_y_id + 1) * CHUNK_TILE_Y_SIZE:
            return
    
    chunk_tile_y_element_ids = chunk_tile_y_id * CHUNK_TILE_Y_SIZE + tl.arange(0, CHUNK_TILE_Y_SIZE)
    chunk_tile_x_element_ids = chunk_tile_x_id * CHUNK_TILE_X_SIZE + tl.arange(0, CHUNK_TILE_X_SIZE)
    state_tile_element_ids = tl.arange(0, STATE_TILE_SIZE)

    # Advance pointers to the current batch, chunk, and group
    B_ptr += batch_id * B_batch_stride + chunk_id * chunk_size * B_seq_stride + group_id * B_group_stride
    C_ptr += batch_id * C_batch_stride + chunk_id * chunk_size * C_seq_stride + group_id * C_group_stride
    CB_ptr += batch_id * CB_batch_stride + chunk_id * CB_chunk_stride + group_id * CB_group_stride

    C_ptrs = C_ptr + (chunk_tile_y_element_ids[:, None] * C_seq_stride + state_tile_element_ids[None, :] * C_state_element_stride)
    B_ptrs = B_ptr + (state_tile_element_ids[:, None] * B_state_element_stride + chunk_tile_x_element_ids[None, :] * B_seq_stride)
    CB_ptrs = CB_ptr + (chunk_tile_y_element_ids[:, None] * CB_chunk_y_element_stride + chunk_tile_x_element_ids[None, :] * CB_chunk_x_element_stride)

    # Compute CB = C @ B^T (or B @ C^T depending on the convention)
    CB = tl.zeros((CHUNK_TILE_Y_SIZE, CHUNK_TILE_X_SIZE), dtype=tl.float32)
    for state_tile_id in range(0, tl.cdiv(state_dim, STATE_TILE_SIZE)):
        C = tl.load(
            C_ptrs + state_tile_id * STATE_TILE_SIZE * C_state_element_stride, 
            mask=(chunk_tile_y_element_ids[:, None] < chunk_size_limit) & (state_tile_element_ids[None, :] < state_dim - state_tile_id * STATE_TILE_SIZE), other=0.0
        )
        B = tl.load(
            B_ptrs + state_tile_id * STATE_TILE_SIZE * B_state_element_stride, 
            mask=(state_tile_element_ids[:, None] < state_dim - state_tile_id * STATE_TILE_SIZE) & (chunk_tile_x_element_ids[None, :] < chunk_size_limit), other=0.0
        )
        CB += tl.dot(C, B)
    
    tl.store(CB_ptrs, CB, mask=(chunk_tile_y_element_ids[:, None] < chunk_size) & (chunk_tile_x_element_ids[None, :] < chunk_size))


####################################################################################################


@triton.autotune(
    configs=[
        triton.Config({'CHUNK_TILE_SIZE': 128, 'HEAD_TILE_SIZE': 256, 'STATE_TILE_SIZE': 64, 'CHUNK_REDUCE_SIZE': 64}, num_stages=3, num_warps=8),
        triton.Config({'CHUNK_TILE_SIZE': 64, 'HEAD_TILE_SIZE': 256, 'STATE_TILE_SIZE': 32, 'CHUNK_REDUCE_SIZE': 32}, num_stages=4, num_warps=4),
        triton.Config({'CHUNK_TILE_SIZE': 128, 'HEAD_TILE_SIZE': 128, 'STATE_TILE_SIZE': 32, 'CHUNK_REDUCE_SIZE': 32}, num_stages=4, num_warps=4),
        triton.Config({'CHUNK_TILE_SIZE': 128, 'HEAD_TILE_SIZE': 64, 'STATE_TILE_SIZE': 32, 'CHUNK_REDUCE_SIZE': 32}, num_stages=4, num_warps=4),
        triton.Config({'CHUNK_TILE_SIZE': 64, 'HEAD_TILE_SIZE': 128, 'STATE_TILE_SIZE': 32, 'CHUNK_REDUCE_SIZE': 32}, num_stages=4, num_warps=4),
        triton.Config({'CHUNK_TILE_SIZE': 128, 'HEAD_TILE_SIZE': 64, 'STATE_TILE_SIZE': 64, 'CHUNK_REDUCE_SIZE': 64}, num_stages=4, num_warps=4),
        triton.Config({'CHUNK_TILE_SIZE': 64, 'HEAD_TILE_SIZE': 128, 'STATE_TILE_SIZE': 64, 'CHUNK_REDUCE_SIZE': 64}, num_stages=4, num_warps=4),
        triton.Config({'CHUNK_TILE_SIZE': 128, 'HEAD_TILE_SIZE': 32, 'STATE_TILE_SIZE': 32, 'CHUNK_REDUCE_SIZE': 32}, num_stages=4, num_warps=4),
        triton.Config({'CHUNK_TILE_SIZE': 64, 'HEAD_TILE_SIZE': 32, 'STATE_TILE_SIZE': 32, 'CHUNK_REDUCE_SIZE': 32}, num_stages=5, num_warps=2),
        triton.Config({'CHUNK_TILE_SIZE': 32, 'HEAD_TILE_SIZE': 64, 'STATE_TILE_SIZE': 32, 'CHUNK_REDUCE_SIZE': 32}, num_stages=5, num_warps=2),
        triton.Config({'CHUNK_TILE_SIZE': 64, 'HEAD_TILE_SIZE': 64, 'STATE_TILE_SIZE': 32, 'CHUNK_REDUCE_SIZE': 32}, num_stages=4, num_warps=2),
    ],
    key=['chunk_size', 'head_dim', 'state_dim', 'IS_CAUSAL'],
)
@triton.jit
def chunk_scan_forward_kernel(
    u_ptr,  # (batch_size, seq_len, num_heads, head_dim)
    delta_ptr,  # (batch_size, num_heads, num_chunks, chunk_size)
    decay_cumsum_ptr,   # (batch_size, num_heads, num_chunks, chunk_size)
    C_ptr,  # (batch_size, seq_len, num_groups, state_dim)
    h_ptr,  # (batch_size, num_chunks, num_heads, head_dim, state_dim)
    CB_ptr, # (batch_size, num_chunks, num_groups, chunk_size, chunk_size)
    y_ptr,  # (batch_size, seq_len, num_heads, head_dim)

    batch_size, seq_len, chunk_size, head_dim, state_dim, num_heads_per_group,

    u_batch_stride, u_seq_stride, u_head_stride, u_head_element_stride,
    delta_batch_stride, delta_head_stride, delta_chunk_stride, delta_chunk_element_stride,
    decay_cumsum_batch_stride, decay_cumsum_head_stride, decay_cumsum_chunk_stride, decay_cumsum_chunk_element_stride,
    C_batch_stride, C_seq_stride, C_group_stride, C_state_element_stride,
    h_batch_stride, h_chunk_stride, h_head_stride, h_head_element_stride, h_state_element_stride,
    CB_batch_stride, CB_chunk_stride, CB_group_stride, CB_chunk_y_element_stride, CB_chunk_x_element_stride,
    y_batch_stride, y_seq_stride, y_head_stride, y_head_element_stride,

    IS_CAUSAL: tl.constexpr,
    CHUNK_TILE_SIZE: tl.constexpr,
    HEAD_TILE_SIZE: tl.constexpr,
    STATE_TILE_SIZE: tl.constexpr,
    CHUNK_REDUCE_SIZE: tl.constexpr,
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
    chunk_size_limit = min(chunk_size, seq_len - chunk_id * chunk_size)

    chunk_tile_element_ids = chunk_tile_id * CHUNK_TILE_SIZE + tl.arange(0, CHUNK_TILE_SIZE)
    head_tile_element_ids = head_tile_id * HEAD_TILE_SIZE + tl.arange(0, HEAD_TILE_SIZE)
    chunk_reduce_element_ids = tl.arange(0, CHUNK_REDUCE_SIZE)
    state_tile_element_ids = tl.arange(0, STATE_DIM_ALIGNED if STATE_DIM_ALIGNED <= 128 else STATE_TILE_SIZE)

    # Advance raw pointers to the current batch, chunk, head, and group
    u_ptr += batch_id * u_batch_stride + chunk_id * chunk_size * u_seq_stride + head_id * u_head_stride
    delta_ptr += batch_id * delta_batch_stride + chunk_id * delta_chunk_stride + head_id * delta_head_stride
    decay_cumsum_ptr += batch_id * decay_cumsum_batch_stride + chunk_id * decay_cumsum_chunk_stride + head_id * decay_cumsum_head_stride
    C_ptr += batch_id * C_batch_stride + chunk_id * chunk_size * C_seq_stride + (head_id // num_heads_per_group) * C_group_stride
    h_ptr += batch_id * h_batch_stride + chunk_id * h_chunk_stride + head_id * h_head_stride
    CB_ptr += batch_id * CB_batch_stride + chunk_id * CB_chunk_stride + (head_id // num_heads_per_group) * CB_group_stride
    y_ptr += batch_id * y_batch_stride + chunk_id * chunk_size * y_seq_stride + head_id * y_head_stride

    # Set up tile pointers
    u_ptrs = u_ptr + (chunk_reduce_element_ids[:, None] * u_seq_stride + head_tile_element_ids[None, :] * u_head_element_stride)    
    delta_ptrs = delta_ptr + chunk_reduce_element_ids * delta_chunk_element_stride
    decay_cumsum_ptrs = decay_cumsum_ptr + chunk_tile_element_ids * decay_cumsum_chunk_element_stride    
    decay_cumsum_inter_ptrs = decay_cumsum_ptr + chunk_reduce_element_ids * decay_cumsum_chunk_element_stride    
    C_ptrs = C_ptr + (chunk_tile_element_ids[:, None] * C_seq_stride + state_tile_element_ids[None, :] * C_state_element_stride)    
    h_ptrs = h_ptr + (head_tile_element_ids[None, :] * h_head_element_stride + state_tile_element_ids[:, None] * h_state_element_stride)    
    CB_ptrs = CB_ptr + (chunk_tile_element_ids[:, None] * CB_chunk_y_element_stride + chunk_reduce_element_ids[None, :] * CB_chunk_x_element_stride)
    y_ptrs = y_ptr + (chunk_tile_element_ids[:, None] * y_seq_stride + head_tile_element_ids[None, :] * y_head_element_stride)

    # Contribution from the initial state: y = (C @ h) * exp(decay_cumsum)
    decay_cumsum = tl.load(decay_cumsum_ptrs, mask=chunk_tile_element_ids < chunk_size, other=0.0)
    y = tl.zeros((CHUNK_TILE_SIZE, HEAD_TILE_SIZE), dtype=tl.float32)
    scale = tl.exp(decay_cumsum)
    
    if STATE_DIM_ALIGNED <= 128:
        C = tl.load(C_ptrs, mask=(chunk_tile_element_ids[:, None] < chunk_size_limit) & (state_tile_element_ids[None, :] < state_dim), other=0.0)
        h = tl.load(h_ptrs, mask=(state_tile_element_ids[:, None] < state_dim) & (head_tile_element_ids[None, :] < head_dim), other=0.0)
        y = tl.dot(C, h) * scale[:, None]
    else:
        for state_tile_id in range(1 + (state_dim - 1) // STATE_TILE_SIZE):
            C = tl.load(C_ptrs + state_tile_id * STATE_TILE_SIZE, mask=(chunk_tile_element_ids[:, None] < chunk_size_limit) & (state_tile_element_ids[None, :] < state_dim - state_tile_id * STATE_TILE_SIZE))
            h = tl.load(h + state_tile_id * STATE_TILE_SIZE, mask=(state_tile_element_ids[:, None] < state_dim - state_tile_id * STATE_TILE_SIZE) & (head_tile_element_ids[None, :] < head_dim), other=0.0)
            y += tl.dot(C, h)
        y *= scale[:, None]
    
    # Contribution from interactions within the chunk: y += (CB * decay_inter) @ u
    causal_tile_sequence_bound = chunk_size_limit if not IS_CAUSAL else min((chunk_tile_id + 1) * CHUNK_TILE_SIZE, chunk_size_limit)
    for chunk_reduce_id in range(1 + (causal_tile_sequence_bound - 1) // CHUNK_REDUCE_SIZE):
        CB = tl.load(
            CB_ptrs + chunk_reduce_id * CHUNK_REDUCE_SIZE * CB_chunk_x_element_stride, 
            mask=(chunk_tile_element_ids[:, None] < chunk_size) & (chunk_reduce_element_ids[None, :] < chunk_size - chunk_reduce_id * CHUNK_REDUCE_SIZE), other=0.0
        )
        decay_cumsum_inter = tl.load(
            decay_cumsum_inter_ptrs + chunk_reduce_id * CHUNK_REDUCE_SIZE * decay_cumsum_chunk_element_stride, 
            mask=chunk_reduce_element_ids < chunk_size - chunk_reduce_id * CHUNK_REDUCE_SIZE, other=0.0
        )
        CB *= tl.exp(tl.minimum((decay_cumsum[:, None] - decay_cumsum_inter[None, :]), 0.0))
        delta_inter = tl.load(
            delta_ptrs + chunk_reduce_id * CHUNK_REDUCE_SIZE * delta_chunk_element_stride,
            mask=chunk_reduce_element_ids < chunk_size - chunk_reduce_id * CHUNK_REDUCE_SIZE, other=0.0
        )
        CB *= delta_inter
        if IS_CAUSAL:
            CB = tl.where(
                chunk_tile_element_ids[:, None] >= chunk_reduce_id * CHUNK_REDUCE_SIZE + chunk_reduce_element_ids[None, :],
                CB, 0.0
            )
        u = tl.load(
            u_ptrs + chunk_reduce_id * CHUNK_REDUCE_SIZE * u_seq_stride,
            mask=(chunk_reduce_element_ids[:, None] < chunk_size_limit - chunk_reduce_id * CHUNK_REDUCE_SIZE) & (head_tile_element_ids[None, :] < head_dim), other=0.0
        )
        y += tl.dot(CB, u)

    tl.store(y_ptrs, y, mask=(chunk_tile_element_ids[:, None] < chunk_size_limit) & (head_tile_element_ids[None, :] < head_dim))