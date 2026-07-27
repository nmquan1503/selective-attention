import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_D": 32}, num_warps=1),
        triton.Config({"BLOCK_D": 64}, num_warps=2),
        triton.Config({"BLOCK_D": 128}, num_warps=4),
        triton.Config({"BLOCK_D": 256}, num_warps=8),
    ],
    key=["D"],
)
@triton.jit
def append_kv_cache_kernel(
    k_ptr,                  # (num_seqs, num_heads, dim)
    v_ptr,                  # (num_seqs, num_heads, dim)
    log_gate_ptr,           # (num_seqs, num_heads)
    k_cache_ptr,            # (total_keys, dim)
    v_cache_ptr,            # (total_keys, dim)
    log_gate_cache_ptr,     # (total_keys,)
    write_pos_ptr,          # (num_groups,)
    k_t_stride, k_h_stride, k_d_stride,
    log_gate_t_stride, log_gate_h_stride,
    k_cache_k_stride, k_cache_d_stride,
    NUM_HEADS: tl.constexpr,
    D: tl.constexpr,
    BLOCK_D: tl.constexpr
):
    seq_id = tl.program_id(0)
    head_id = tl.program_id(1)
    d_block_id = tl.program_id(2)

    group_id = seq_id * NUM_HEADS + head_id
    write_pos = tl.load(write_pos_ptr + group_id)

    offs_d = d_block_id * BLOCK_D + tl.arange(0, BLOCK_D)
    mask_d = offs_d < D

    k_ptrs = (
        k_ptr
        + seq_id * k_t_stride
        + head_id * k_h_stride
        + offs_d * k_d_stride
    )
    k = tl.load(k_ptrs, mask=mask_d, other=0.0)

    v_ptrs = (
        v_ptr
        + seq_id * k_t_stride
        + head_id * k_h_stride
        + offs_d * k_d_stride
    )
    v = tl.load(v_ptrs, mask=mask_d, other=0.0)

    k_cache_ptrs = (
        k_cache_ptr
        + write_pos * k_cache_k_stride
        + offs_d * k_cache_d_stride
    )
    v_cache_ptrs = (
        v_cache_ptr
        + write_pos * k_cache_k_stride
        + offs_d * k_cache_d_stride
    )

    tl.store(k_cache_ptrs, k, mask=mask_d)
    tl.store(v_cache_ptrs, v, mask=mask_d)

    if d_block_id == 0:
        log_gate = tl.load(
            log_gate_ptr
            + seq_id * log_gate_t_stride
            + head_id * log_gate_h_stride
        )
        tl.store(log_gate_cache_ptr + write_pos, log_gate)



@triton.autotune(
    configs=[
        triton.Config({"BLOCK_D": 32}, num_warps=2),
        triton.Config({"BLOCK_D": 32}, num_warps=4),

        triton.Config({"BLOCK_D": 64}, num_warps=2),
        triton.Config({"BLOCK_D": 64}, num_warps=4),
        triton.Config({"BLOCK_D": 64}, num_warps=8),

        triton.Config({"BLOCK_D": 128}, num_warps=4),
        triton.Config({"BLOCK_D": 128}, num_warps=8),

        triton.Config({"BLOCK_D": 256}, num_warps=8),
    ],
    key=["D", "BLOCK_T"],
)
@triton.jit
def pad_buffer_kernel(
    k_cache_ptr,            # (total_keys, dim)
    v_cache_ptr,            # (total_keys, dim)
    log_gate_cache_ptr,     # (total_keys,)
    k_out_ptr,              # (new_total_keys, dim)
    v_out_ptr,              # (new_total_keys, dim)
    log_gate_out_ptr,       # (new_total_keys,)
    cu_seqlens_k_ptr,       # (num_groups + 1,)
    new_cu_seqlens_k_ptr,   # (num_groups + 1,)
    write_pos_ptr,          # (num_groups,)
    pad_offsets_ptr,        # (num_groups + 1,)
    k_cache_k_stride, k_cache_d_stride,
    k_out_k_stride, k_out_d_stride,
    NUM_GROUPS: tl.constexpr,
    D: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_D: tl.constexpr
):
    token_block_id = tl.program_id(0)
    d_block_id = tl.program_id(1)

    lo = 0
    hi = NUM_GROUPS

    while lo < hi:
        mid = (lo + hi) // 2

        if tl.load(pad_offsets_ptr + mid) <= token_block_id:
            lo = mid + 1
        else:
            hi = mid

    group_id = lo - 1
    group_block_id = token_block_id - tl.load(pad_offsets_ptr + group_id)

    src_start = tl.load(cu_seqlens_k_ptr + group_id)
    src_end = tl.load(write_pos_ptr + group_id)
    dst_start = tl.load(new_cu_seqlens_k_ptr + group_id)

    offs_t = group_block_id * BLOCK_T + tl.arange(0, BLOCK_T)
    mask_t = offs_t < (src_end - src_start)

    offs_d = d_block_id * BLOCK_D + tl.arange(0, BLOCK_D)
    mask_d = offs_d < D

    mask = mask_t[:, None] & mask_d[None, :]

    k_ptrs = (
        k_cache_ptr 
        + (src_start + offs_t[:, None]) * k_cache_k_stride 
        + offs_d[None, :] * k_cache_d_stride
    )
    k = tl.load(k_ptrs, mask=mask, other=0.0)

    k_out_ptrs = (
        k_out_ptr
        + (dst_start + offs_t[:, None]) * k_out_k_stride
        + offs_d[None, :] * k_out_d_stride
    )
    tl.store(k_out_ptrs, k, mask=mask)

    v_ptrs = (
        v_cache_ptr
        + (src_start + offs_t[:, None]) * k_cache_k_stride
        + offs_d[None, :] * k_cache_d_stride
    )
    v = tl.load(v_ptrs, mask=mask, other=0.0)

    v_out_ptrs = (
        v_out_ptr
        + (dst_start + offs_t[:, None]) * k_out_k_stride
        + offs_d[None, :] * k_out_d_stride
    )
    tl.store(v_out_ptrs, v, mask=mask)

    if d_block_id == 0:
        log_gate = tl.load(
            log_gate_cache_ptr + src_start + offs_t,
            mask=mask_t, other=0.0,
        )
        tl.store(
            log_gate_out_ptr + dst_start + offs_t,
            log_gate,
            mask=mask_t,
        )



@triton.autotune(
    configs=[
        triton.Config({"BLOCK_D": 32}, num_warps=2),

        triton.Config({"BLOCK_D": 64}, num_warps=2),
        triton.Config({"BLOCK_D": 64}, num_warps=4),

        triton.Config({"BLOCK_D": 128}, num_warps=4),
        triton.Config({"BLOCK_D": 128}, num_warps=8),

        triton.Config({"BLOCK_D": 256}, num_warps=8),
    ],
    key=["D", "BLOCK_T"],
)
@triton.jit
def qk_matmul_kernel(
    q_ptr,                  # (num_seqs, num_heads, D)
    k_cache_ptr,            # (total_keys, D)
    log_gate_cache_ptr,     # (total_keys,)
    cu_seqlens_k_ptr,       # (num_groups + 1,)
    write_pos_ptr,          # (num_groups,)
    scores_ptr,             # (total_keys,)
    cu_seqlens_scores_ptr,  # (num_groups + 1,)
    qk_offsets_ptr,         # (num_groups + 1,)
    q_t_stride, q_h_stride, q_d_stride,
    k_k_stride, k_d_stride,
    NUM_GROUPS: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    SCALE: tl.constexpr,
    D: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    global_block_id = tl.program_id(0)

    lo = 0
    hi = NUM_GROUPS

    while lo < hi:
        mid = (lo + hi) // 2

        if tl.load(qk_offsets_ptr + mid) <= global_block_id:
            lo = mid + 1
        else:
            hi = mid

    group_id = lo - 1
    group_block_id = global_block_id - tl.load(qk_offsets_ptr + group_id)

    seq_id = group_id // NUM_HEADS
    head_id = group_id % NUM_HEADS

    src_start = tl.load(cu_seqlens_k_ptr + group_id)
    src_end = tl.load(write_pos_ptr + group_id) + 1
    seq_len = src_end - src_start

    dst_start = tl.load(cu_seqlens_scores_ptr + group_id)

    offs_t = group_block_id * BLOCK_T + tl.arange(0, BLOCK_T)
    mask_t = offs_t < seq_len
    mask_t_past = offs_t < src_end - src_start - 1

    offs_d_base = tl.arange(0, BLOCK_D)

    acc = tl.zeros((BLOCK_T,), dtype=tl.float32)

    for d_start in range(0, D, BLOCK_D):
        offs_d = d_start + offs_d_base
        mask_d = offs_d < D

        q_ptrs = (
            q_ptr
            + seq_id * q_t_stride
            + head_id * q_h_stride
            + offs_d * q_d_stride
        )
        q = tl.load(q_ptrs, mask=mask_d, other=0.0)

        k_ptrs = (
            k_cache_ptr
            + (src_start + offs_t[:, None]) * k_k_stride
            + offs_d[None, :] * k_d_stride
        )
        k = tl.load(
            k_ptrs,
            mask=mask_t[:, None] & mask_d[None, :],
            other=0.0,
        )

        acc += tl.sum(k * q[None, :], axis=1)

    acc = acc * SCALE

    log_gate = tl.load(
        log_gate_cache_ptr + src_start + offs_t,
        mask=mask_t_past,
        other=0.0,
    )
    scores = acc + log_gate

    scores_ptrs = scores_ptr + dst_start + offs_t
    tl.store(scores_ptrs, scores, mask=mask_t)



@triton.jit
def softmax_kernel(
    scores_ptr,             # (total_keys,)
    cu_seqlens_scores_ptr,  # (num_groups + 1,)
    scores_stride,
    BLOCK_T: tl.constexpr,
):
    group_id = tl.program_id(0)

    start = tl.load(cu_seqlens_scores_ptr + group_id)
    end = tl.load(cu_seqlens_scores_ptr + group_id + 1)
    seq_len = end - start

    offs_t_base = tl.arange(0, BLOCK_T)

    max_score = -float("inf")

    for block_start in range(0, seq_len, BLOCK_T):
        offs_t = block_start + offs_t_base
        mask = offs_t < seq_len

        scores = tl.load(
            scores_ptr + (start + offs_t) * scores_stride,
            mask=mask,
            other=-float("inf"),
        )

        max_score = tl.maximum(max_score, tl.max(scores, axis=0))

    denom = 0.0

    for block_start in range(0, seq_len, BLOCK_T):
        offs_t = block_start + offs_t_base
        mask = offs_t < seq_len

        scores = tl.load(
            scores_ptr + (start + offs_t) * scores_stride,
            mask=mask,
            other=-float("inf"),
        )

        denom += tl.sum(tl.exp(scores - max_score), axis=0)

    for block_start in range(0, seq_len, BLOCK_T):
        offs_t = block_start + offs_t_base
        mask = offs_t < seq_len

        scores_ptrs = scores_ptr + (start + offs_t) * scores_stride
        scores = tl.load(scores_ptrs, mask=mask, other=0.0)

        weights = tl.exp(scores - max_score) / denom

        tl.store(scores_ptrs, weights, mask=mask)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_D": 32}, num_warps=2),
        triton.Config({"BLOCK_D": 32}, num_warps=4),

        triton.Config({"BLOCK_D": 64}, num_warps=2),
        triton.Config({"BLOCK_D": 64}, num_warps=4),

        triton.Config({"BLOCK_D": 128}, num_warps=4),
        triton.Config({"BLOCK_D": 128}, num_warps=8),

        triton.Config({"BLOCK_D": 256}, num_warps=8),
    ],
    key=["D", "BLOCK_T"],
)
@triton.jit
def attn_v_kernel(
    scores_ptr,             # (total_keys,)
    v_cache_ptr,            # (total_keys, D)
    cu_seqlens_scores_ptr,  # (num_groups + 1,)
    cu_seqlens_k_ptr,       # (num_groups + 1,)
    output_ptr,             # (num_seqs, num_heads, dim)
    v_cache_k_stride, v_cache_d_stride,
    output_t_stride, output_h_stride, output_d_stride,
    scores_stride,
    NUM_HEADS: tl.constexpr,
    D: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    group_id = tl.program_id(0)
    d_block_id = tl.program_id(1)

    seq_id = group_id // NUM_HEADS
    head_id = group_id % NUM_HEADS

    scores_start = tl.load(cu_seqlens_scores_ptr + group_id)
    scores_end = tl.load(cu_seqlens_scores_ptr + group_id + 1)
    seq_len = scores_end - scores_start

    v_start = tl.load(cu_seqlens_k_ptr + group_id)

    offs_d = d_block_id * BLOCK_D + tl.arange(0, BLOCK_D)
    mask_d = offs_d < D

    offs_t_base = tl.arange(0, BLOCK_T)

    acc = tl.zeros((BLOCK_D,), dtype=tl.float32)

    for block_start in range(0, seq_len, BLOCK_T):
        offs_t = block_start + offs_t_base
        mask_t = offs_t < seq_len

        scores_ptrs = scores_ptr + (scores_start + offs_t) * scores_stride
        scores = tl.load(scores_ptrs, mask=mask_t, other=0.0)

        v_ptrs = (
            v_cache_ptr
            + (v_start + offs_t[:, None]) * v_cache_k_stride
            + offs_d[None, :] * v_cache_d_stride
        )
        v = tl.load(
            v_ptrs,
            mask=mask_t[:, None] & mask_d[None, :],
            other=0.0,
        )

        acc += tl.sum(scores[:, None] * v, axis=0)

    output_ptrs = (
        output_ptr
        + seq_id * output_t_stride
        + head_id * output_h_stride
        + offs_d * output_d_stride
    )
    tl.store(output_ptrs, acc, mask=mask_d)