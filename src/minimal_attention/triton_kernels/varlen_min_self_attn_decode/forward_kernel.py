import triton
import triton.language as tl

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



@triton.jit
def min_attn_decode_kernel(
    q_ptr,              # (num_seqs, num_heads, dim)
    k_ptr,              # (num_seqs, num_heads, dim)
    v_ptr,              # (num_seqs, num_heads, dim)
    gate_ptr,           # (num_seqs, num_heads)
    scores_std_ptr,     # (num_heads,)
    gate_threshold_ptr, # (num_heads,)
    k_cache_ptr,        # (total_keys, dim)
    v_cache_ptr,        # (total_keys, dim)
    log_gate_cache_ptr, # (total_keys,)
    cu_seqlens_k_ptr,   # (num_groups + 1,)
    write_pos_ptr,      # (num_groups,)
    mid_o_ptr,          # (num_groups, num_chunks, dim)
    mid_logsumexp_ptr,  # (num_groups, num_chunks)
    q_t_stride, q_h_stride, q_d_stride,
    gate_t_stride, gate_h_stride,
    k_cache_k_stride, k_cache_d_stride,
    mid_o_g_stride, mid_o_s_stride, mid_o_d_stride,
    mid_logsumexp_g_stride, mid_logsumexp_s_stride,
    NUM_HEADS: tl.constexpr,
    SCALE: tl.constexpr,
    D: tl.constexpr,
    HAS_GATE_THRESHOLD: tl.constexpr,
    LOG_GATE_PENALTY: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
    BLOCK_K: tl.constexpr
):
    seq_id = tl.program_id(0)
    head_id = tl.program_id(1)
    chunk_id = tl.program_id(2)
    group_id = seq_id * NUM_HEADS + head_id

    k_start = tl.load(cu_seqlens_k_ptr + group_id)
    write_pos = tl.load(write_pos_ptr + group_id)

    chunk_start = k_start + chunk_id * CHUNK_SIZE
    chunk_end = tl.minimum(chunk_start + CHUNK_SIZE, write_pos)
    chunk_len = chunk_end - chunk_start

    if chunk_len < 0:
        return

    offs_d = tl.arange(0, D)

    q = tl.load(
        q_ptr
        + seq_id * q_t_stride
        + head_id * q_h_stride
        + offs_d[None, :] * q_d_stride
    )

    if chunk_len == CHUNK_SIZE:
        acc = tl.zeros((D,), dtype=tl.float32)
        m = tl.full((), float("-inf"), dtype=tl.float32)
        l = tl.full((), 0.0, dtype=tl.float32)

    else:
        k = tl.load(
            k_ptr
            + seq_id * q_t_stride
            + head_id * q_h_stride
            + offs_d[:, None] * q_d_stride
        )

        v = tl.load(
            v_ptr
            + seq_id * q_t_stride
            + head_id * q_h_stride
            + offs_d * q_d_stride
        )

        m = tl.reshape(tl.dot(q, k), ()).to(tl.float32) * SCALE
        l = 1.0
        acc = v

        gate = tl.load(
            gate_ptr
            + seq_id * gate_t_stride
            + head_id * gate_h_stride
        )

        if HAS_GATE_THRESHOLD:
            threshold = tl.load(gate_threshold_ptr + head_id)

            if gate > threshold and gate > 0:
                scores_std = tl.load(scores_std_ptr + head_id)

                tl.store(
                    log_gate_cache_ptr + chunk_end,
                    tl.log(gate) * scores_std * LOG_GATE_PENALTY,
                )

                tl.store(
                    k_cache_ptr
                    + chunk_end * k_cache_k_stride
                    + offs_d * k_cache_d_stride,
                    tl.reshape(k, (D,)),
                )

                tl.store(
                    v_cache_ptr
                    + chunk_end * k_cache_k_stride
                    + offs_d * k_cache_d_stride,
                    v,
                )

                tl.store(
                    write_pos_ptr + group_id,
                    write_pos + 1,
                )

        else:
            scores_std = tl.load(scores_std_ptr + head_id)

            tl.store(
                log_gate_cache_ptr + chunk_end,
                tl.log(gate) * scores_std * LOG_GATE_PENALTY,
            )

            tl.store(
                k_cache_ptr
                + chunk_end * k_cache_k_stride
                + offs_d * k_cache_d_stride,
                tl.reshape(k, (D,)),
            )

            tl.store(
                v_cache_ptr
                + chunk_end * k_cache_k_stride
                + offs_d * k_cache_d_stride,
                v,
            )

            tl.store(
                write_pos_ptr + group_id,
                write_pos + 1,
            )

    offs_cache_base = tl.arange(0, BLOCK_K)

    for block_start in range(0, chunk_len, BLOCK_K):
        offs_t = chunk_start + block_start + offs_cache_base
        mask_t = offs_t < chunk_end

        k_cache_ptrs = (
            k_cache_ptr
            + offs_t[:, None] * k_cache_k_stride
            + offs_d[None, :] * k_cache_d_stride
        )
        k_cache = tl.load(k_cache_ptrs, mask=mask_t[:, None], other=0.0)

        log_gate = tl.load(
            log_gate_cache_ptr + offs_t,
            mask=mask_t,
            other=float("-inf"),
        )

        scores = (
            tl.reshape(
                tl.dot(q, tl.trans(k_cache)),
                (BLOCK_K,),
            )
            * SCALE
            + log_gate
        )

        cur_max = tl.max(scores)
        new_max = tl.maximum(m, cur_max)

        alpha = tl.exp(m - new_max)
        p = tl.exp(scores - new_max)

        v_cache_ptrs = (
            v_cache_ptr
            + offs_t[:, None] * k_cache_k_stride
            + offs_d[None, :] * k_cache_d_stride
        )
        v_cache = tl.load(v_cache_ptrs, mask=mask_t[:, None], other=0.0)

        acc = acc * alpha + tl.reshape(
            tl.dot(p[None, :], v_cache),
            (D,),
        )

        l = l * alpha + tl.sum(p, axis=0)
        m = new_max

    mid_o = acc / l

    tl.store(
        mid_o_ptr
        + group_id * mid_o_g_stride
        + chunk_id * mid_o_s_stride
        + offs_d * mid_o_d_stride,
        mid_o,
    )

    tl.store(
        mid_logsumexp_ptr
        + group_id * mid_logsumexp_g_stride
        + chunk_id * mid_logsumexp_s_stride,
        m + tl.log(l),
    )

@triton.jit
def reduce_kernel(
    gate_ptr,               # (num_seqs, num_heads)
    gate_threshold_ptr,     # (num_heads,)
    mid_o_ptr,              # (num_groups, num_chunks, dim)
    mid_logsumexp_ptr,      # (num_groups, num_chunks)
    cu_seqlens_k_ptr,       # (num_groups + 1,)
    write_pos_ptr,          # (num_groups,)
    out_ptr,                # (num_seqs, num_heads, dim)
    gate_t_stride, gate_h_stride,
    mid_o_g_stride, mid_o_c_stride, mid_o_d_stride,
    mid_logsumexp_g_stride, mid_logsumexp_c_stride,
    out_t_stride, out_h_stride, out_d_stride,
    NUM_HEADS: tl.constexpr,
    D: tl.constexpr,
    HAS_GATE_THRESHOLD: tl.constexpr,
    CHUNK_SIZE: tl.constexpr
):
    seq_id = tl.program_id(0)
    head_id = tl.program_id(1)
    group_id = seq_id * NUM_HEADS + head_id

    k_start = tl.load(cu_seqlens_k_ptr + group_id)
    write_pos = tl.load(write_pos_ptr + group_id)

    cache_len = write_pos - k_start

    gate = tl.load(
        gate_ptr
        + seq_id * gate_t_stride
        + head_id * gate_h_stride
    )

    if HAS_GATE_THRESHOLD:
        threshold = tl.load(gate_threshold_ptr + head_id)
        if not (gate > threshold and gate > 0):
            cache_len += 1

    num_chunks = tl.cdiv(cache_len, CHUNK_SIZE)

    offs_d = tl.arange(0, D)

    acc = tl.zeros((D,), dtype=tl.float32)
    m = tl.full((), float("-inf"), dtype=tl.float32)
    l = tl.full((), 0.0, dtype=tl.float32)

    for chunk_id in range(num_chunks):
        cur_l = tl.load(
            mid_logsumexp_ptr
            + group_id * mid_logsumexp_g_stride
            + chunk_id * mid_logsumexp_c_stride
        )

        cur_o = tl.load(
            mid_o_ptr
            + group_id * mid_o_g_stride
            + chunk_id * mid_o_c_stride
            + offs_d * mid_o_d_stride
        )

        new_m = tl.maximum(m, cur_l)

        alpha = tl.exp(m - new_m)
        weight = tl.exp(cur_l - new_m)

        acc = acc * alpha + cur_o * weight
        l = l * alpha + weight
        m = new_m

    out = acc / l

    tl.store(
        out_ptr
        + seq_id * out_t_stride
        + head_id * out_h_stride
        + offs_d * out_d_stride,
        out,
    )