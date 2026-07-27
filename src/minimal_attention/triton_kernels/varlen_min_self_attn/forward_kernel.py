import torch
import triton
import triton.language as tl

@triton.jit
def count_kept_kernel(
    valid_ptr,          # (total_tokens, num_heads)
    cu_seqlens_ptr,     # (num_seqs + 1,)
    counts_ptr,         # (num_seqs, num_heads)
    valid_t_stride, valid_h_stride,
    NUM_HEADS: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    seq_id = tl.program_id(0)
    head_id = tl.program_id(1)

    seq_start = tl.load(cu_seqlens_ptr + seq_id)
    seq_end = tl.load(cu_seqlens_ptr + seq_id + 1)
    seq_len = seq_end - seq_start

    count = 0
    for block_start in range(0, seq_len, BLOCK_T):
        offs_t = block_start + tl.arange(0, BLOCK_T)
        mask = offs_t < seq_len

        valid = tl.load(
            valid_ptr
            + (seq_start + offs_t) * valid_t_stride
            + head_id * valid_h_stride,
            mask=mask,
            other=0,
        )
        count += tl.sum(valid.to(tl.int32), axis=0)

    counts_ptr = counts_ptr + seq_id * NUM_HEADS + head_id
    tl.store(counts_ptr, count)



@triton.jit
def compact_1_kernel(
    log_gate_ptr,       # (total_tokens, num_heads)
    valid_ptr,          # (total_tokens, num_heads)
    cu_seqlens_ptr,     # (num_seqs + 1,)
    cu_seqlens_k_ptr,   # (num_groups + 1,)
    log_gate_out_ptr,   # (total_keys,)
    k_ids_out_ptr,      # (total_keys,)
    log_gate_t_stride, log_gate_h_stride,
    NUM_HEADS: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    seq_id = tl.program_id(0)
    head_id = tl.program_id(1)

    seq_start = tl.load(cu_seqlens_ptr + seq_id)
    seq_end = tl.load(cu_seqlens_ptr + seq_id + 1)
    seq_len = seq_end - seq_start

    group_id = seq_id * NUM_HEADS + head_id
    out_start = tl.load(cu_seqlens_k_ptr + group_id)

    offs_t_base = tl.arange(0, BLOCK_T)
    out_idx = 0

    for block_start in range(0, seq_len, BLOCK_T):
        offs_t = block_start + offs_t_base
        mask_t = offs_t < seq_len
        input_pos = seq_start + offs_t

        valid = tl.load(
            valid_ptr
            + input_pos * log_gate_t_stride
            + head_id * log_gate_h_stride,
            mask=mask_t,
            other=False,
        )
        log_gate = tl.load(
            log_gate_ptr
            + input_pos * log_gate_t_stride
            + head_id * log_gate_h_stride,
            mask=mask_t,
            other=0.0,
        )

        valid_i32 = valid.to(tl.int32)
        rank = tl.cumsum(valid_i32, axis=0) - 1
        out_pos = out_start + out_idx + rank

        tl.store(log_gate_out_ptr + out_pos, log_gate, mask=valid)
        tl.store(k_ids_out_ptr + out_pos, input_pos, mask=valid)

        out_idx += tl.sum(valid_i32, axis=0)



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
def compact_2_kernel(
    k_ptr,              # (total_tokens, num_heads, dim)
    v_ptr,              # (total_tokens, num_heads, dim)
    k_ids_ptr,          # (total_keys,)
    cu_seqlens_k_ptr,   # (num_groups + 1,)
    k_out_ptr,          # (total_keys, dim)
    v_out_ptr,          # (total_keys, dim)
    k_t_stride, k_h_stride, k_d_stride,
    NUM_HEADS: tl.constexpr,
    D: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    seq_id = tl.program_id(0)
    head_id = tl.program_id(1)
    d_block_id = tl.program_id(2)

    group_id = seq_id * NUM_HEADS + head_id
    out_start = tl.load(cu_seqlens_k_ptr + group_id)
    out_end = tl.load(cu_seqlens_k_ptr + group_id + 1)
    num_keys = out_end - out_start

    d_start = d_block_id * BLOCK_D
    offs_d = d_start + tl.arange(0, BLOCK_D)
    mask_d = offs_d < D

    offs_t_base = tl.arange(0, BLOCK_T)
    for block_start in range(0, num_keys, BLOCK_T):
        offs_t = block_start + offs_t_base
        out_pos = out_start + offs_t
        mask_t = out_pos < out_end

        original_idx = tl.load(k_ids_ptr + out_pos, mask=mask_t, other=0)

        k_ptrs = (
            k_ptr
            + original_idx[:, None] * k_t_stride
            + head_id * k_h_stride
            + offs_d[None, :] * k_d_stride
        )

        v_ptrs = (
            v_ptr
            + original_idx[:, None] * k_t_stride
            + head_id * k_h_stride
            + offs_d[None, :] * k_d_stride
        )

        k_out_ptrs = (
            k_out_ptr
            + out_pos[:, None] * D
            + offs_d[None, :]
        )

        v_out_ptrs = (
            v_out_ptr
            + out_pos[:, None] * D
            + offs_d[None, :]
        )

        mask = mask_t[:, None] & mask_d[None, :]

        k_val = tl.load(k_ptrs, mask=mask, other=0.0)
        v_val = tl.load(v_ptrs, mask=mask, other=0.0)

        tl.store(k_out_ptrs, k_val, mask=mask)
        tl.store(v_out_ptrs, v_val, mask=mask)



@triton.autotune(
    configs=[
        triton.Config({"BLOCK_D": 32}, num_warps=2),
        triton.Config({"BLOCK_D": 32}, num_warps=4),

        triton.Config({"BLOCK_D": 64}, num_warps=2),
        triton.Config({"BLOCK_D": 64}, num_warps=4),
        triton.Config({"BLOCK_D": 64}, num_warps=8),

        triton.Config({"BLOCK_D": 128}, num_warps=4),
        triton.Config({"BLOCK_D": 128}, num_warps=8),
    ],
    key=["D", "BLOCK_Q", "BLOCK_K"],
)
@triton.jit
def qk_matmul_kernel(
    q_ptr,                  # (total_tokens, num_heads, dim)
    k_ptr,                  # (total_keys, dim)
    log_gate_ptr,           # (total_keys,)
    k_ids_ptr,              # (total_keys,)
    scores_ptr,             # (total_scores,)
    cu_seqlens_ptr,         # (num_seqs + 1,)
    cu_seqlens_k_ptr,       # (num_groups + 1,)
    cu_seqlens_scores_ptr,  # (num_groups + 1,)
    qk_offsets_ptr,         # (num_groups + 1,)
    q_lens_ptr,             # (num_groups,)
    k_lens_ptr,             # (num_groups,)
    q_t_stride, q_h_stride, q_d_stride,
    k_k_stride, k_d_stride,
    NUM_GROUPS: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    D: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_K: tl.constexpr,
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

    q_len = tl.load(q_lens_ptr + group_id)
    k_len = tl.load(k_lens_ptr + group_id)

    num_k_blocks = tl.cdiv(k_len, BLOCK_K)
    q_block_id = group_block_id // num_k_blocks
    k_block_id = group_block_id % num_k_blocks

    seq_id = group_id // NUM_HEADS
    head_id = group_id % NUM_HEADS

    q_start = tl.load(cu_seqlens_ptr + seq_id)
    k_start = tl.load(cu_seqlens_k_ptr + group_id)

    offs_q = q_block_id * BLOCK_Q + tl.arange(0, BLOCK_Q)
    offs_k = k_block_id * BLOCK_K + tl.arange(0, BLOCK_K)
    offs_d_base = tl.arange(0, BLOCK_D)

    q_mask = offs_q < q_len
    k_mask = offs_k < k_len

    scores = tl.zeros((BLOCK_Q, BLOCK_K), dtype=tl.float32)

    for d_start in range(0, D, BLOCK_D):
        offs_d = d_start + offs_d_base
        d_mask = offs_d < D

        q_ptrs = (
            q_ptr
            + (q_start + offs_q[:, None]) * q_t_stride
            + head_id * q_h_stride
            + offs_d[None, :] * q_d_stride
        )
        q = tl.load(
            q_ptrs,
            mask=q_mask[:, None] & d_mask[None, :],
            other=0.0,
        )

        k_ptrs = (
            k_ptr
            + (k_start + offs_k[:, None]) * k_k_stride
            + offs_d[None, :] * k_d_stride
        )
        k = tl.load(
            k_ptrs,
            mask=k_mask[:, None] & d_mask[None, :],
            other=0.0,
        )

        scores += tl.dot(q, tl.trans(k))

    log_gate = tl.load(
        log_gate_ptr + k_start + offs_k,
        mask=k_mask,
        other=0.0,
    )
    scores += log_gate[None, :]

    q_global_idx = q_start + offs_q
    k_global_idx = tl.load(
        k_ids_ptr + k_start + offs_k,
        mask=k_mask,
        other=-1,
    )

    if IS_CAUSAL:
        allowed_mask = (
            k_global_idx[None, :]
            < q_global_idx[:, None]
        )
    else:
        allowed_mask = (
            k_global_idx[None, :]
            != q_global_idx[:, None]
        )

    scores = tl.where(allowed_mask, scores, float("-inf"))

    scores_start = tl.load(cu_seqlens_scores_ptr + group_id)
    output_width = k_len + 1

    scores_ptrs = (
        scores_ptr
        + scores_start
        + offs_q[:, None] * output_width
        + offs_k[None, :]
        + 1
    )

    scores_mask = q_mask[:, None] & k_mask[None, :]
    tl.store(scores_ptrs, scores, mask=scores_mask)



@triton.jit
def fill_self_score_kernel(
    self_score_ptr,         # (total_tokens, num_heads)
    scores_ptr,             # (total_scores,)
    cu_seqlens_ptr,         # (num_seqs + 1,)
    cu_seqlens_scores_ptr,  # (num_groups + 1,)
    fill_block_offsets_ptr, # (num_groups + 1,)
    q_lens_ptr,             # (num_groups,)
    k_lens_ptr,             # (num_groups,)
    self_t_stride, self_h_stride,
    NUM_GROUPS: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    BLOCK_Q: tl.constexpr,
):
    global_block_id = tl.program_id(0)
    lo = 0
    hi = NUM_GROUPS
    while lo < hi:
        mid = (lo + hi) // 2
        if tl.load(fill_block_offsets_ptr + mid) <= global_block_id:
            lo = mid + 1
        else:
            hi = mid

    group_id = lo - 1
    group_block_id = global_block_id - tl.load(fill_block_offsets_ptr + group_id)

    q_len = tl.load(q_lens_ptr + group_id)
    k_len = tl.load(k_lens_ptr + group_id)
    output_width = k_len + 1

    seq_id = group_id // NUM_HEADS
    head_id = group_id % NUM_HEADS
    q_start = tl.load(cu_seqlens_ptr + seq_id)

    offs_q = group_block_id * BLOCK_Q + tl.arange(0, BLOCK_Q)
    q_mask = offs_q < q_len

    self_score_ptrs = (
        self_score_ptr
        + (q_start + offs_q) * self_t_stride
        + head_id * self_h_stride
    )
    self_score = tl.load(
        self_score_ptrs,
        mask=q_mask,
        other=0.0,
    )

    scores_start = tl.load(cu_seqlens_scores_ptr + group_id)
    scores_ptrs = (
        scores_ptr
        + scores_start
        + offs_q * output_width
    )

    tl.store(scores_ptrs, self_score, mask=q_mask)



@triton.jit
def softmax_kernel(
    scores_ptr,                     # (total_scores,)
    cu_seqlens_scores_ptr,          # (num_groups + 1,)
    softmax_block_offsets_ptr,      # (num_groups + 1,)
    q_lens_ptr,                     # (num_groups,)
    k_lens_ptr,                     # (num_groups,)
    NUM_GROUPS: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    global_block_id = tl.program_id(0)

    lo, hi = 0, NUM_GROUPS
    while lo < hi:
        mid = (lo + hi) // 2
        if tl.load(softmax_block_offsets_ptr + mid) <= global_block_id:
            lo = mid + 1
        else:
            hi = mid

    group_id = lo - 1
    group_block_id = global_block_id - tl.load(softmax_block_offsets_ptr + group_id)

    q_len = tl.load(q_lens_ptr + group_id)
    k_len = tl.load(k_lens_ptr + group_id)
    output_width = k_len + 1 
    scores_start = tl.load(cu_seqlens_scores_ptr + group_id)

    offs_q = group_block_id * BLOCK_Q + tl.arange(0, BLOCK_Q)
    q_mask = offs_q < q_len

    row_max = tl.full((BLOCK_Q,), value=float("-inf"), dtype=tl.float32)
    for k_start_local in range(0, output_width, BLOCK_K):
        offs_k = k_start_local + tl.arange(0, BLOCK_K)
        k_mask = offs_k < output_width

        scores_ptrs = (
            scores_ptr
            + scores_start
            + offs_q[:, None] * output_width
            + offs_k[None, :]
        )
        scores = tl.load(
            scores_ptrs,
            mask=q_mask[:, None] & k_mask[None, :],
            other=-float("inf"),
        )

        block_max = tl.max(scores, axis=1)
        row_max = tl.maximum(row_max, block_max)

    row_sum = tl.zeros((BLOCK_Q,), dtype=tl.float32)
    for k_start_local in range(0, output_width, BLOCK_K):
        offs_k = k_start_local + tl.arange(0, BLOCK_K)
        k_mask = offs_k < output_width

        scores_ptrs = (
            scores_ptr
            + scores_start
            + offs_q[:, None] * output_width
            + offs_k[None, :]
        )
        scores = tl.load(
            scores_ptrs,
            mask=q_mask[:, None] & k_mask[None, :],
            other=-float("inf"),
        )

        exp_scores = tl.exp(scores - row_max[:, None])
        row_sum += tl.sum(exp_scores, axis=1)

    inv_row_sum = 1.0 / row_sum
    for k_start_local in range(0, output_width, BLOCK_K):
        offs_k = k_start_local + tl.arange(0, BLOCK_K)
        k_mask = offs_k < output_width

        scores_ptrs = (
            scores_ptr
            + scores_start
            + offs_q[:, None] * output_width
            + offs_k[None, :]
        )
        scores = tl.load(
            scores_ptrs,
            mask=q_mask[:, None] & k_mask[None, :],
            other=float("-inf"),
        )

        probs = tl.exp(scores - row_max[:, None]) * inv_row_sum[:, None]
        tl.store(scores_ptrs, probs, mask=(q_mask[:, None] & k_mask[None, :]))



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
    key=["D", "BLOCK_Q", "BLOCK_K"],
)
@triton.jit
def attn_output_kernel(
    scores_ptr,                 # (total_scores,)
    original_v_ptr,             # (total_tokens, num_heads, dim)
    v_ptr,                      # (total_keys, dim)
    out_ptr,                    # (total_tokens, num_heads, dim)
    cu_seqlens_ptr,             # (num_seqs + 1,)
    cu_seqlens_k_ptr,           # (num_groups + 1,)
    cu_seqlens_scores_ptr,      # (num_groups + 1,)
    attn_block_offsets_ptr,     # (num_groups + 1,)
    q_lens_ptr,                 # (num_groups,)
    k_lens_ptr,                 # (num_groups,)
    original_v_t_stride, original_v_h_stride, original_v_d_stride,
    v_t_stride, v_d_stride,
    out_t_stride, out_h_stride, out_d_stride,
    NUM_GROUPS: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    D: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    global_block_id = tl.program_id(0)
    d_block_id = tl.program_id(1)

    lo, hi = 0, NUM_GROUPS
    while lo < hi:
        mid = (lo + hi) // 2
        if tl.load(attn_block_offsets_ptr + mid) <= global_block_id:
            lo = mid + 1
        else:
            hi = mid

    group_id = lo - 1
    group_block_id = global_block_id - tl.load(attn_block_offsets_ptr + group_id)

    q_len = tl.load(q_lens_ptr + group_id)
    k_len = tl.load(k_lens_ptr + group_id)
    scores_start = tl.load(cu_seqlens_scores_ptr + group_id)

    seq_id = group_id // NUM_HEADS
    head_id = group_id % NUM_HEADS
    q_start = tl.load(cu_seqlens_ptr + seq_id)
    k_start = tl.load(cu_seqlens_k_ptr + group_id)

    offs_q = group_block_id * BLOCK_Q + tl.arange(0, BLOCK_Q)
    q_mask = offs_q < q_len

    offs_d = d_block_id * BLOCK_D + tl.arange(0, BLOCK_D)
    d_mask = offs_d < D

    acc = tl.zeros((BLOCK_Q, BLOCK_D), dtype=tl.float32)

    output_width = k_len + 1
    self_score_ptrs = (
        scores_ptr
        + scores_start
        + offs_q * output_width
    )
    self_prob = tl.load(
        self_score_ptrs,
        mask=q_mask,
        other=0.0,
    )

    original_v_ptrs = (
        original_v_ptr
        + (q_start + offs_q[:, None]) * original_v_t_stride
        + head_id * original_v_h_stride
        + offs_d[None, :] * original_v_d_stride
    )
    original_v = tl.load(
        original_v_ptrs,
        mask=(q_mask[:, None] & d_mask[None, :]), other=0.0
    )
    acc += self_prob[:, None] * original_v

    for k_block_start in range(0, k_len, BLOCK_K):
        offs_k = k_block_start + tl.arange(0, BLOCK_K)
        k_mask = offs_k < k_len

        score_ptrs = (
            scores_ptr
            + scores_start
            + offs_q[:, None] * output_width
            + offs_k[None, :]
            + 1
        )
        probs = tl.load(
            score_ptrs,
            mask=q_mask[:, None] & k_mask[None, :],
            other=0.0,
        )

        v_ptrs = (
            v_ptr
            + (k_start + offs_k[:, None]) * v_t_stride
            + offs_d[None, :] * v_d_stride
        )
        v_tile = tl.load(
            v_ptrs,
            mask=k_mask[:, None] & d_mask[None, :],
            other=0.0,
        )
        
        acc += tl.dot(probs, v_tile)

    out_ptrs = (
        out_ptr
        + (q_start + offs_q[:, None]) * out_t_stride
        + head_id * out_h_stride
        + offs_d[None, :] * out_d_stride
    )
    tl.store(out_ptrs, acc, mask=(q_mask[:, None] & d_mask[None, :]))


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
def pack_sequences_kernel(
    x_ptr,                  # (batch_size, seq_len, dim)
    packed_ptr,             # (total_tokens, dim)
    cu_seqlens_ptr,         # (batch_size + 1,)
    pack_offsets_ptr,       # (batch_size + 1,)
    x_b_stride, x_s_stride, x_d_stride,
    packed_t_stride, packed_d_stride,
    B: tl.constexpr,
    D: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    block_id = tl.program_id(0)
    d_block_id = tl.program_id(1)

    lo, hi = 0, B

    while lo < hi:
        mid = (lo + hi) // 2

        if tl.load(pack_offsets_ptr + mid) <= block_id:
            lo = mid + 1
        else:
            hi = mid

    b = lo - 1
    t_block = block_id - tl.load(pack_offsets_ptr + b)

    seq_start = tl.load(cu_seqlens_ptr + b)
    seq_len = tl.load(cu_seqlens_ptr + b + 1) - seq_start

    t = t_block * BLOCK_T + tl.arange(0, BLOCK_T)
    d = d_block_id * BLOCK_D + tl.arange(0, BLOCK_D)

    mask = (t[:, None] < seq_len) & (d[None, :] < D)

    x_ptrs = (
        x_ptr
        + b * x_b_stride
        + t[:, None] * x_s_stride
        + d[None, :] * x_d_stride
    )

    x = tl.load(x_ptrs, mask=mask, other=0.0)

    packed_ptrs = (
        packed_ptr
        + (seq_start + t[:, None]) * packed_t_stride
        + d[None, :] * packed_d_stride
    )

    tl.store(packed_ptrs, x, mask=mask)



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
def unpack_sequences_kernel(
    packed_ptr,             # (total_tokens, dim)
    x_ptr,                  # (batch_size, max_seq_len, dim)
    cu_seqlens_ptr,         # (batch_size + 1,)
    unpack_offsets_ptr,     # (batch_size + 1,)
    packed_t_stride, packed_d_stride,
    x_b_stride, x_s_stride, x_d_stride,
    B: tl.constexpr,
    D: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    block_id = tl.program_id(0)
    d_block_id = tl.program_id(1)

    lo, hi = 0, B
    while lo < hi:
        mid = (lo + hi) // 2

        if tl.load(unpack_offsets_ptr + mid) <= block_id:
            lo = mid + 1
        else:
            hi = mid

    b = lo - 1
    t_block = block_id - tl.load(unpack_offsets_ptr + b)

    seq_start = tl.load(cu_seqlens_ptr + b)
    seq_len = tl.load(cu_seqlens_ptr + b + 1) - seq_start

    offs_t = t_block * BLOCK_T + tl.arange(0, BLOCK_T)
    offs_d = d_block_id * BLOCK_D + tl.arange(0, BLOCK_D)

    mask = (offs_t[:, None] < seq_len) & (offs_d[None, :] < D)

    packed_ptrs = (
        packed_ptr
        + (seq_start + offs_t[:, None]) * packed_t_stride
        + offs_d[None, :] * packed_d_stride
    )
    val = tl.load(packed_ptrs, mask=mask, other=0.0)

    x_ptrs = (
        x_ptr
        + b * x_b_stride
        + offs_t[:, None] * x_s_stride
        + offs_d[None, :] * x_d_stride
    )
    tl.store(x_ptrs, val, mask=mask)



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
def unpack_sequences_kernel(
    packed_ptr,             # (total_tokens, dim)
    x_ptr,                  # (batch_size, max_seq_len, dim)
    cu_seqlens_ptr,         # (batch_size + 1,)
    unpack_offsets_ptr,     # (batch_size + 1,)
    packed_t_stride, packed_d_stride,
    x_b_stride, x_s_stride, x_d_stride,
    B: tl.constexpr,
    D: tl.constexpr,
    MAX_SEQ_LEN: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    block_id = tl.program_id(0)
    d_block_id = tl.program_id(1)

    lo, hi = 0, B
    while lo < hi:
        mid = (lo + hi) // 2

        if tl.load(unpack_offsets_ptr + mid) <= block_id:
            lo = mid + 1
        else:
            hi = mid

    b = lo - 1
    t_block = block_id - tl.load(unpack_offsets_ptr + b)

    seq_start = tl.load(cu_seqlens_ptr + b)
    seq_len = tl.load(cu_seqlens_ptr + b + 1) - seq_start

    offs_t = t_block * BLOCK_T + tl.arange(0, BLOCK_T)
    offs_d = d_block_id * BLOCK_D + tl.arange(0, BLOCK_D)

    mask = (offs_t[:, None] < seq_len) & (offs_d[None, :] < D)

    packed_ptrs = (
        packed_ptr
        + (seq_start + offs_t[:, None]) * packed_t_stride
        + offs_d[None, :] * packed_d_stride
    )
    val = tl.load(packed_ptrs, mask=mask, other=0.0)

    x_ptrs = (
        x_ptr
        + b * x_b_stride
        + offs_t[:, None] * x_s_stride
        + offs_d[None, :] * x_d_stride
    )
    tl.store(x_ptrs, val, mask=mask)