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
def compact_kernel(
    k_ptr,              # (total_tokens, num_heads, D)
    v_ptr,              # (total_tokens, num_heads, D)
    log_gate_ptr,       # (total_tokens, num_heads)
    valid_ptr,          # (total_tokens, num_heads)
    cu_seqlens_ptr,     # (num_seqs + 1)
    cu_seqlens_k_ptr,   # (num_groups + 1)
    k_out_ptr,          # (total_keys, D)
    v_out_ptr,          # (total_keys, D)
    log_gate_out_ptr,   # (total_keys)
    k_ids_out_ptr,      # (total_keys)
    k_t_stride, k_h_stride, k_d_stride,
    log_gate_t_stride, log_gate_h_stride,
    NUM_HEADS: tl.constexpr,
    D: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    seq_id = tl.program_id(0)
    head_id = tl.program_id(1)
    d_block_id = tl.program_id(2)

    seq_start = tl.load(cu_seqlens_ptr + seq_id)
    seq_len = tl.load(cu_seqlens_ptr + seq_id + 1) - seq_start

    group_id = seq_id * NUM_HEADS + head_id
    out_start = tl.load(cu_seqlens_k_ptr + group_id)

    offs_t_base = tl.arange(0, BLOCK_T)
    offs_d = d_block_id * BLOCK_D + tl.arange(0, BLOCK_D)
    mask_d = offs_d < D

    out_idx = 0

    for block_start in range(0, seq_len, BLOCK_T):
        offs_t = block_start + offs_t_base
        input_pos = seq_start + offs_t
        mask_t = offs_t < seq_len

        valid = tl.load(
            valid_ptr + input_pos * log_gate_t_stride + head_id * log_gate_h_stride,
            mask=mask_t, other=False,
        )

        valid_i32 = valid.to(tl.int32)
        out_pos = out_start + out_idx + tl.cumsum(valid_i32, axis=0) - 1
        out_idx += tl.sum(valid_i32, axis=0)

        mask = mask_t[:, None] & valid[:, None] & mask_d[None, :]

        k_ptrs = k_ptr + input_pos[:, None] * k_t_stride + head_id * k_h_stride + offs_d[None, :] * k_d_stride
        v_ptrs = v_ptr + input_pos[:, None] * k_t_stride + head_id * k_h_stride + offs_d[None, :] * k_d_stride

        k_out_ptrs = k_out_ptr + out_pos[:, None] * D + offs_d[None, :]
        v_out_ptrs = v_out_ptr + out_pos[:, None] * D + offs_d[None, :]

        tl.store(k_out_ptrs, tl.load(k_ptrs, mask=mask, other=0.0), mask=mask)
        tl.store(v_out_ptrs, tl.load(v_ptrs, mask=mask, other=0.0), mask=mask)

        if d_block_id == 0:
            log_gate = tl.load(
                log_gate_ptr + input_pos * log_gate_t_stride + head_id * log_gate_h_stride,
                mask=mask_t, other=0.0,
            )
            tl.store(log_gate_out_ptr + out_pos, log_gate, mask=valid)
            tl.store(k_ids_out_ptr + out_pos, input_pos, mask=valid)



@triton.jit
def min_self_attn_kernel(
    q_ptr,              # (total_tokens, num_heads, dim)
    k_ptr,              # (total_keys, dim)
    v_ptr,              # (total_keys, dim)
    log_gate_ptr,       # (total_keys,)
    k_ids_ptr,          # (total_keys,)
    self_score_ptr,     # (total_tokens, num_heads)
    orig_v_ptr,         # (total_tokens, num_heads, dim)
    out_ptr,            # (total_tokens, num_heads, dim)
    cu_seqlens_ptr,     # (num_seqs + 1,)
    cu_seqlens_k_ptr,   # (num_groups + 1,)
    q_t_stride, q_h_stride, q_d_stride,
    k_k_stride, k_d_stride,
    v_k_stride, v_d_stride,
    orig_v_t_stride, orig_v_h_stride, orig_v_d_stride,
    out_t_stride, out_h_stride, out_d_stride,
    self_t_stride, self_h_stride,
    NUM_HEADS: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    SCALE: tl.constexpr,
    D: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    seq_id = tl.program_id(0)
    head_id = tl.program_id(1)
    q_block_id = tl.program_id(2)
    group_id = seq_id * NUM_HEADS + head_id

    q_start = tl.load(cu_seqlens_ptr + seq_id)
    q_end = tl.load(cu_seqlens_ptr + seq_id + 1)
    q_len = q_end - q_start

    if q_block_id * BLOCK_Q >= q_len:
        return

    k_start = tl.load(cu_seqlens_k_ptr + group_id)
    k_end = tl.load(cu_seqlens_k_ptr + group_id + 1)
    k_len = k_end - k_start

    offs_q = q_block_id * BLOCK_Q + tl.arange(0, BLOCK_Q)
    offs_d = tl.arange(0, D)

    mask_q = offs_q < q_len
    q_ids = q_start + offs_q

    q = tl.load(
        q_ptr
        + q_ids[:, None] * q_t_stride
        + head_id * q_h_stride
        + offs_d[None, :] * q_d_stride,
        mask=mask_q[:, None],
        other=0.0,
    )

    self_score = tl.load(
        self_score_ptr
        + q_ids * self_t_stride
        + head_id * self_h_stride,
        mask=mask_q,
        other=-float("inf"),
    ) * SCALE

    row_max = self_score
    row_sum_exp = tl.exp(self_score - row_max)

    orig_v = tl.load(
        orig_v_ptr
        + q_ids[:, None] * orig_v_t_stride
        + head_id * orig_v_h_stride
        + offs_d[None, :] * orig_v_d_stride,
        mask=mask_q[:, None],
        other=0.0,
    )

    acc = row_sum_exp[:, None] * orig_v

    for k_block_start in range(0, k_len, BLOCK_K):
        offs_k = k_block_start + tl.arange(0, BLOCK_K)
        mask_k = offs_k < k_len
        k_abs = k_start + offs_k

        k = tl.load(
            k_ptr
            + k_abs[:, None] * k_k_stride
            + offs_d[None, :] * k_d_stride,
            mask=mask_k[:, None],
            other=0.0,
        )

        scores = tl.dot(q, tl.trans(k)) * SCALE

        log_gate = tl.load(log_gate_ptr + k_abs, mask=mask_k, other=float("-inf"))
        scores += log_gate[None, :]

        k_ids = tl.load(k_ids_ptr + k_abs, mask=mask_k, other=-1)

        if IS_CAUSAL:
            allowed = k_ids[None, :] < q_ids[:, None]
        else:
            allowed = k_ids[None, :] != q_ids[:, None]

        scores = tl.where(allowed, scores, -float("inf"))

        block_max = tl.max(scores, axis=1)
        new_max = tl.maximum(row_max, block_max)
        alpha = tl.exp(row_max - new_max)

        exp_scores = tl.exp(scores - new_max[:, None])
        block_sum = tl.sum(exp_scores, axis=1)

        acc *= alpha[:, None]

        v = tl.load(
            v_ptr
            + k_abs[:, None] * v_k_stride
            + offs_d[None, :] * v_d_stride,
            mask=mask_k[:, None],
            other=0.0,
        )

        acc += tl.dot(exp_scores, v)
        row_sum_exp = row_sum_exp * alpha + block_sum
        row_max = new_max

    acc /= row_sum_exp[:, None]

    tl.store(
        out_ptr
        + q_ids[:, None] * out_t_stride
        + head_id * out_h_stride
        + offs_d[None, :] * out_d_stride,
        acc,
        mask=mask_q[:, None],
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