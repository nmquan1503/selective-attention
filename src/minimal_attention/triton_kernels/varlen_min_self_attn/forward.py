import torch
import triton

from .forward_kernel import (
    count_kept_kernel, compact_1_kernel, compact_2_kernel,
    qk_matmul_kernel, fill_self_score_kernel,
    softmax_kernel,
    attn_output_kernel,
    pack_sequences_kernel, unpack_sequences_kernel
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
    avg_seqlen = total_tokens // num_seqs
    block_t = min(256, triton.next_power_of_2(avg_seqlen))

    counts = torch.empty(
        (num_seqs, num_heads), 
        device=k.device, dtype=torch.int32
    )
    count_kept_kernel[(num_seqs, num_heads)](
        valid, cu_seqlens, counts,
        *(valid.stride()),
        BLOCK_T=max(block_t, 32),
        NUM_HEADS=num_heads,
        num_warps=max(1, block_t // 32)
    )

    counts_flat = counts.reshape(-1)
    cu_seqlens_k = torch.empty(
        num_seqs * num_heads + 1, 
        device=k.device, dtype=torch.int32
    )
    cu_seqlens_k[0] = 0
    cu_seqlens_k[1:] = torch.cumsum(counts_flat, dim=0)

    total_keys = cu_seqlens_k[-1].item()
    k_out = torch.empty(
        (total_keys, dim), 
        device=k.device, dtype=k.dtype
    )
    v_out = torch.empty(
        (total_keys, dim), 
        device=v.device, dtype=v.dtype
    )
    log_gate_out = torch.empty(
        total_keys, 
        device=log_gate.device, dtype=log_gate.dtype
    )
    k_ids = torch.empty(
        total_keys, 
        device=k.device, dtype=torch.int32
    )
    compact_1_kernel[
        (num_seqs, num_heads)
    ](
        log_gate, valid,
        cu_seqlens, cu_seqlens_k,
        log_gate_out, k_ids,
        *(log_gate.stride()),
        NUM_HEADS=num_heads,
        BLOCK_T=max(block_t, 32),
        num_warps=max(1, block_t // 32)
    )

    grid = lambda META: (
        num_seqs, 
        num_heads, 
        triton.cdiv(dim, META["BLOCK_D"])
    )
    compact_2_kernel[grid](
        k, v, k_ids,
        cu_seqlens_k,
        k_out, v_out,
        *(k.stride()),
        NUM_HEADS=num_heads,
        D=dim,
        BLOCK_T=block_t
    )

    return k_out, v_out, log_gate_out, k_ids, cu_seqlens_k

def _qk_matmul(
    q: torch.Tensor,
    k: torch.Tensor,
    log_gate: torch.Tensor,
    k_ids: torch.Tensor,
    self_score: torch.Tensor,
    cu_seqlens: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    q_lens: torch.Tensor,
    k_lens: torch.Tensor,
    avg_q_len: int,
    avg_k_len: int,
    is_causal: bool,
):
    """
    Args:
        q: (total_tokens, num_heads, dim)
        k: (total_keys, dim)
        log_gate, k_ids: (total_keys,)
        self_score: (total_tokens, num_heads)
        cu_seqlens: (num_seqs + 1,)
        cu_seqlens_k: (num_groups + 1,)
        q_lens, k_lens: (num_groups + 1,)
    
    Returns:
        scores: (total_scores,)
        cu_seqlens_scores: (num_seqs * num_heads + 1,)
    """
    total_q_tokens, num_heads, dim = q.shape
    num_seqs = cu_seqlens.numel() - 1
    num_groups = num_seqs * num_heads

    scores_lens = q_lens * (k_lens + 1)
    cu_seqlens_scores = torch.zeros(
        num_groups + 1, 
        device=q.device, dtype=torch.int32
    )
    cu_seqlens_scores[1:] = torch.cumsum(scores_lens, dim=0)

    total_scores = cu_seqlens_scores[-1].item()
    scores = torch.empty(
        total_scores, 
        device=q.device, dtype=q.dtype
    )

    if avg_q_len <= 32:
        block_q_matmul = 16
    elif avg_q_len <= 128:
        block_q_matmul = 32
    else:
        block_q_matmul = 64

    if avg_k_len <= 32:
        block_k_matmul = 16
    elif avg_k_len <= 128:
        block_k_matmul = 32
    else:
        block_k_matmul = 64

    num_q_blocks = triton.cdiv(q_lens, block_q_matmul)
    num_k_blocks = triton.cdiv(k_lens, block_k_matmul)
    blocks_per_group = num_q_blocks * num_k_blocks

    qk_offsets = torch.zeros(
        num_groups + 1, 
        device=q.device, 
        dtype=torch.int32
    )
    qk_offsets[1:] = torch.cumsum(blocks_per_group, dim=0)
    total_qk_blocks = qk_offsets[-1].item()

    qk_matmul_kernel[(total_qk_blocks,)](
        q, k, log_gate, k_ids, scores,
        cu_seqlens, cu_seqlens_k, cu_seqlens_scores,
        qk_offsets, q_lens, k_lens,
        *q.stride(),
        *k.stride(),
        NUM_GROUPS=num_groups,
        NUM_HEADS=num_heads,
        IS_CAUSAL=is_causal,
        D=dim,
        BLOCK_Q=block_q_matmul,
        BLOCK_K=block_k_matmul,
    )

    block_q_fill = min(256, max(triton.next_power_of_2(avg_q_len), 32))
    num_q_blocks_fill = triton.cdiv(q_lens, block_q_fill)

    fill_offsets = torch.zeros(
        num_groups + 1, 
        device=q.device, dtype=torch.int32
    )
    fill_offsets[1:] = torch.cumsum(num_q_blocks_fill, dim=0)
    total_fill_blocks = fill_offsets[-1].item()

    fill_self_score_kernel[(total_fill_blocks,)](
        self_score, scores,
        cu_seqlens, cu_seqlens_scores,
        fill_offsets, q_lens, k_lens,
        *(self_score.stride()),
        NUM_GROUPS=num_groups,
        NUM_HEADS=num_heads,
        BLOCK_Q=block_q_fill,
        num_warps=max(1, block_q_fill // 32)
    )

    return scores, cu_seqlens_scores

def _softmax(
    scores: torch.Tensor,
    cu_seqlens_scores: torch.Tensor,
    q_lens: torch.Tensor,
    k_lens: torch.Tensor,
    avg_q_len: int,
    avg_k_len: int,
    num_groups: int
):
    """
    Args:
        scores: (total_scores,)
        cu_seqlens_scores: (num_groups + 1,)
        q_lens, k_lens: (num_groups,)
    
    Returns:
        scores: (total_scores,)
    """
    if avg_q_len <= 64:
        block_q = 32
    elif avg_q_len <= 256:
        block_q = 64
    elif avg_q_len <= 1024:
        block_q = 128
    else:
        block_q = 256

    if avg_k_len <= 32:
        block_k = 32
    elif avg_k_len <= 128:
        block_k = 64
    elif avg_k_len <= 512:
        block_k = 128
    else:
        block_k = 256

    max_tile_elements = 8192
    block_q = max(1, min(block_q, max_tile_elements // block_k))

    num_q_blocks = triton.cdiv(q_lens, block_q)
    softmax_offsets = torch.zeros(
        num_groups + 1,
        device=scores.device,
        dtype=torch.int32,
    )
    softmax_offsets[1:] = torch.cumsum(num_q_blocks, dim=0)
    total_softmax_blocks = softmax_offsets[-1].item()

    if block_q <= 32:
        num_warps = 2
    elif block_q <= 64:
        num_warps = 4
    else:
        num_warps = 8

    softmax_kernel[(total_softmax_blocks,)](
        scores, cu_seqlens_scores,
        softmax_offsets, q_lens, k_lens,
        NUM_GROUPS=num_groups,
        BLOCK_Q=block_q,
        BLOCK_K=block_k,
        num_warps=num_warps
    )

    return scores

def _attn_output(
    scores: torch.Tensor,
    original_v: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    cu_seqlens_scores: torch.Tensor,
    q_lens: torch.Tensor,
    k_lens: torch.Tensor,
    num_groups: int,
    avg_q_len: int,
    avg_k_len: int,
):
    """
    Args:
        scores: (total_scores,)
        original_v: (total_tokens, num_heads, dim)
        v: (total_keys, dim)
        cu_seqlens: (num_seqs + 1,)
        cu_seqlens_k: (num_groups + 1,)
        cu_seqlens_scores: (num_groups + 1,)
        q_lens, k_lens: (num_groups,)
    
    Returns:
        out: (total_tokens, num_heads, dim)
    """
    total_tokens, num_heads, dim = original_v.shape

    if avg_q_len <= 64:
        block_q = 32
    else:
        block_q = 64

    if avg_k_len <= 32:
        block_k = 32
    elif avg_k_len <= 128:
        block_k = 64
    else:
        block_k = 128

    num_q_blocks_per_group = (q_lens + block_q - 1) // block_q  # (num_groups,)
    attn_offsets = torch.zeros(
        num_groups + 1,
        dtype=torch.int32,
        device=scores.device,
    )
    attn_offsets[1:] = torch.cumsum(num_q_blocks_per_group, dim=0)
    total_q_blocks = attn_offsets[-1].item()

    out = torch.empty_like(original_v)

    grid = lambda META: (
        total_q_blocks, 
        triton.cdiv(dim, META["BLOCK_D"])
    )
    attn_output_kernel[grid](
        scores, original_v, v, out,
        cu_seqlens, cu_seqlens_k, cu_seqlens_scores,
        attn_offsets, q_lens, k_lens,
        *(original_v.stride()),
        *(v.stride()),
        *(out.stride()),
        NUM_GROUPS=num_groups,  
        NUM_HEADS=num_heads,
        D=dim,
        BLOCK_Q=block_q,
        BLOCK_K=block_k,
    )

    return out

def varlen_min_self_attn_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens: torch.Tensor,
    gate: torch.Tensor,
    scores_std: torch.Tensor,
    log_gate_penalty: float,
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

    total_tokens, num_heads, dim = q.shape
    num_seqs = cu_seqlens.numel() - 1
    num_groups = num_seqs * num_heads

    if gate_threshold is not None:
        valid_key = (gate >= gate_threshold) & (gate > 0)
        if not valid_key.any():
            return v, None, None, None, None
    else:
        valid_key = torch.ones_like(gate, dtype=torch.bool)

    self_score = (q.unsqueeze(-2) @ k.unsqueeze(-1)).squeeze(-1).squeeze(-1)

    original_v = v
    log_gate = torch.log(gate) * scores_std[None, :] * log_gate_penalty

    k, v, log_gate, k_ids, cu_seqlens_k = _compress(k, v, log_gate, valid_key, cu_seqlens)

    q_lens = (
        cu_seqlens[1:] - cu_seqlens[:-1]
    ).unsqueeze(1).expand(-1, num_heads).reshape(-1)
    k_lens = cu_seqlens_k[1:] - cu_seqlens_k[:-1]

    avg_q_len = total_tokens // num_seqs
    avg_k_len = int(k_lens[k_lens > 0].float().mean().item())

    scores, cu_seqlens_scores = _qk_matmul(
        q, k, log_gate, k_ids, self_score, 
        cu_seqlens, cu_seqlens_k,
        q_lens, k_lens, avg_q_len, avg_k_len,
        is_causal
    )

    _softmax(
        scores, cu_seqlens_scores, 
        q_lens, k_lens, avg_q_len, avg_k_len,
        num_groups
    )

    out = _attn_output(
        scores, original_v, v, 
        cu_seqlens, cu_seqlens_k, cu_seqlens_scores,
        q_lens, k_lens, num_groups,
        avg_q_len, avg_k_len
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