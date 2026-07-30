import torch
import triton

from .forward_kernel import (
    count_kept_kernel, compact_kernel,
    min_self_attn_kernel,
    pack_sequences_kernel, unpack_sequences_kernel,
)

def _compress(
    k: torch.Tensor,
    v: torch.Tensor,
    log_gate: torch.Tensor,
    valid: torch.Tensor,
    cu_seqlens: torch.Tensor,
):
    """
    Args:
        k, v: (total_tokens, num_heads, dim)
        log_gate, valid: (total_tokens, num_heads)
        cu_seqlens: (num_seqs + 1,)

    Returns:
        k_out, v_out: (total_keys, dim)
        log_gate_out, k_ids: (total_keys,)
        cu_seqlens_k: (num_groups + 1)
    """

    total_tokens, num_heads, dim = k.shape
    num_seqs = cu_seqlens.numel() - 1

    block_t = max(32, min(256, triton.next_power_of_2(total_tokens // num_seqs)))

    counts = torch.empty((num_seqs, num_heads), device=k.device, dtype=torch.int32)
    count_kept_kernel[(num_seqs, num_heads)](
        valid, cu_seqlens, counts,
        *valid.stride(),
        BLOCK_T=block_t,
        NUM_HEADS=num_heads,
        num_warps=max(1, block_t // 32),
    )

    cu_seqlens_k = torch.empty(num_seqs * num_heads + 1, device=k.device, dtype=torch.int32)
    cu_seqlens_k[0] = 0
    cu_seqlens_k[1:] = counts.reshape(-1).cumsum(0)

    total_keys = cu_seqlens_k[-1].item()

    k_out = torch.empty((total_keys, dim), device=k.device, dtype=k.dtype)
    v_out = torch.empty_like(k_out)
    log_gate_out = torch.empty(total_keys, device=k.device, dtype=log_gate.dtype)
    k_ids = torch.empty(total_keys, device=k.device, dtype=torch.int32)

    compact_kernel[lambda META: (
        num_seqs,
        num_heads,
        triton.cdiv(dim, META["BLOCK_D"]),
    )](
        k, v, log_gate, valid,
        cu_seqlens, cu_seqlens_k,
        k_out, v_out, log_gate_out, k_ids,
        *k.stride(),
        *log_gate.stride(),
        NUM_HEADS=num_heads,
        D=dim,
        BLOCK_T=block_t,
    )

    return k_out, v_out, log_gate_out, k_ids, cu_seqlens_k



def _min_self_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    log_gate: torch.Tensor,
    k_ids: torch.Tensor,
    self_score: torch.Tensor,
    original_v: torch.Tensor,
    cu_seqlens: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen: int,
    avg_q_len: int,
    avg_k_len: int,
    scale: float,
    is_causal: bool,
):
    """
    Args:
        q: (total_tokens, num_heads, dim)
        k: (total_keys, dim)
        v: (total_keys, dim)
        log_gate: (total_keys,)
        k_ids: (total_keys,)
        self_score: (total_tokens, num_heads)
        original_v: (total_tokens, num_heads, dim)
        cu_seqlens: (num_seqs + 1,)
        cu_seqlens_k: (num_groups + 1,)
        max_seqlen: maximum query sequence length
        avg_q_len: average query sequence length
        avg_k_len: average key sequence length

    Returns:
        out: (total_tokens, num_heads, dim)
    """

    total_tokens, num_heads, dim = q.shape
    num_seqs = cu_seqlens.numel() - 1

    out = torch.empty_like(q)

    TARGET_WORK = 64 * 1024
    MIN_BLOCK_Q = 16
    MIN_BLOCK_K = 32

    BLOCK_Q = min(128, triton.next_power_of_2(avg_q_len))
    BLOCK_K = min(128, triton.next_power_of_2(avg_k_len))

    budget = TARGET_WORK // dim

    max_block_q = 1 << (max(1, budget // BLOCK_K).bit_length() - 1)
    BLOCK_Q = max(MIN_BLOCK_Q, min(BLOCK_Q, max_block_q))

    if BLOCK_Q == MIN_BLOCK_Q:
        max_block_k = 1 << (max(1, budget // BLOCK_Q).bit_length() - 1)
        BLOCK_K = max(MIN_BLOCK_K, min(BLOCK_K, max_block_k))

    grid = (
        num_seqs,
        num_heads,
        triton.cdiv(max_seqlen, BLOCK_Q),
    )
    min_self_attn_kernel[grid](
        q, k, v, log_gate, k_ids, self_score, original_v,
        out,
        cu_seqlens, cu_seqlens_k,
        *q.stride(),
        *k.stride(),
        *v.stride(),
        *original_v.stride(),
        *out.stride(),
        *self_score.stride(),
        NUM_HEADS=num_heads,
        IS_CAUSAL=is_causal,
        SCALE=scale,
        D=dim,
        BLOCK_Q=BLOCK_Q,
        BLOCK_K=BLOCK_K,
    )

    return out

def varlen_min_self_attn_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens: torch.Tensor,
    max_seqlen: int,
    gate: torch.Tensor,
    scores_std: torch.Tensor,
    log_gate_penalty: float,
    scale: float,
    gate_threshold: torch.Tensor | None = None,
    is_causal: bool = True
):
    """
    Args:
        q: (total_tokens, num_heads, dim)
        k: (total_tokens, num_heads, dim)
        v: (total_tokens, num_heads, dim)
        cu_seqlens: (num_seqs + 1,)
        gate: (total_tokens, num_heads)
        scores_std: (num_heads,)
        gate_threshold: (total_tokens, num_heads)
    
    Returns:
        out: (total_tokens, num_heads, dim)
        k, v: (total_keys, dim)
        log_gate: (total_keys,)
        cu_seqlens_k: (num_groups + 1,)
    """

    total_tokens, num_heads, _ = q.shape
    num_seqs = cu_seqlens.numel() - 1
    num_groups = num_seqs * num_heads

    if gate_threshold is not None:
        valid_key = (gate >= gate_threshold) & (gate > 0)
        if not valid_key.any():
            return v, None, None, None, None
    else:
        valid_key = torch.ones_like(gate, dtype=torch.bool)

    self_score = (
        (q.unsqueeze(-2) @ k.unsqueeze(-1))
        .squeeze(-1)
        .squeeze(-1)
    )

    original_v = v
    log_gate = torch.log(gate) * scores_std[None, :] * log_gate_penalty

    k, v, log_gate, k_ids, cu_seqlens_k = _compress(
        k, v, log_gate, valid_key,
        cu_seqlens,
    )

    total_keys = k.size(0)
    avg_q_len = total_tokens // num_seqs
    avg_k_len = max(1, total_keys // num_groups)

    out = _min_self_attn(
        q, k, v, log_gate, k_ids, self_score, original_v,
        cu_seqlens, cu_seqlens_k,
        max_seqlen, avg_q_len, avg_k_len,
        scale, is_causal,
    )

    return out, k, v, log_gate, cu_seqlens_k

def pack_sequences(
    x: torch.Tensor,
    lengths: torch.Tensor,
):
    """
    Args:
        x: (batch_size, seq_len, dim)
        lengths: (batch_size,)

    Returns:
        packed: (total_tokens, dim)
        cu_seqlens: (batch_size + 1,)
    """
    batch_size, seq_len, dim = x.shape
    lengths = lengths.to(device=x.device, dtype=torch.int32)

    cu_seqlens = torch.cat([
        torch.zeros(1, dtype=torch.int32, device=x.device),
        lengths.cumsum(0),
    ])

    total_tokens = cu_seqlens[-1].item()
    avg_len = total_tokens // max(1, batch_size)
    block_t = min(128, triton.next_power_of_2(max(1, avg_len)))

    num_token_blocks = triton.cdiv(lengths, block_t)

    pack_offsets = torch.zeros(batch_size + 1, dtype=torch.int32, device=x.device)
    pack_offsets[1:] = num_token_blocks.cumsum(0)

    total_blocks = pack_offsets[-1].item()
    packed = torch.empty(total_tokens, dim, dtype=x.dtype, device=x.device)

    grid = lambda META: (
        total_blocks,
        triton.cdiv(dim, META["BLOCK_D"]),
    )

    pack_sequences_kernel[grid](
        x, packed,
        cu_seqlens, pack_offsets,
        *(x.stride()),
        *(packed.stride()),
        B=batch_size,
        D=dim,
        BLOCK_T=block_t,
    )

    return packed, cu_seqlens


def unpack_sequences(
    packed: torch.Tensor,
    cu_seqlens: torch.Tensor,
):
    """
    Args:
        packed: (total_tokens, dim)
        cu_seqlens: (batch_size + 1,)

    Returns:
        x: (batch_size, max_seq_len, dim)
    """
    batch_size = cu_seqlens.numel() - 1
    total_tokens, dim = packed.shape

    lengths = cu_seqlens[1:] - cu_seqlens[:-1]
    max_seq_len = lengths.max().item()

    x = torch.zeros(
        batch_size,
        max_seq_len,
        dim,
        dtype=packed.dtype,
        device=packed.device,
    )

    avg_len = total_tokens // max(1, batch_size)
    block_t = min(128, triton.next_power_of_2(max(1, avg_len)))

    num_token_blocks = triton.cdiv(lengths, block_t)

    unpack_offsets = torch.zeros(
        batch_size + 1,
        dtype=torch.int32,
        device=packed.device,
    )
    unpack_offsets[1:] = num_token_blocks.cumsum(0)

    total_blocks = unpack_offsets[-1].item()

    grid = lambda META: (
        total_blocks,
        triton.cdiv(dim, META["BLOCK_D"]),
    )

    unpack_sequences_kernel[grid](
        packed,
        x,
        cu_seqlens,
        unpack_offsets,
        *packed.stride(),
        *x.stride(),
        B=batch_size,
        D=dim,
        BLOCK_T=block_t,
    )

    return x