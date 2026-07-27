import torch
import triton
import math

from .forward_kernel import (
    append_kv_cache_kernel,
    pad_buffer_kernel,
    qk_matmul_kernel,
    softmax_kernel,
    attn_v_kernel
)

def _append_kv_cache(
    k: torch.Tensor,
    v: torch.Tensor,
    gate: torch.Tensor,
    scores_std: torch.Tensor,
    log_gate_penalty: float,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    log_gate_cache: torch.Tensor,
    write_pos: torch.Tensor,
    gate_threshold: torch.Tensor | None = None,
):
    """
    Args:
        k, v: (num_seqs, num_heads, dim)
        gate: (num_seqs, num_heads)
        scores_std: (num_heads,)
        k_cache, v_cache: (total_keys, dim)
        log_gate_cache: (total_keys,)
        write_pos: (num_groups,)
        gate_threshold: (num_heads,)
    
    Returns:
        valid: (num_seqs, num_heads)
    """
    num_seqs, num_heads, dim = k.shape

    log_gate = torch.log(gate) * scores_std[None, :] * log_gate_penalty
    valid = torch.ones_like(gate, dtype=torch.bool)

    if gate_threshold is not None:
        valid = gate >= gate_threshold[None, :]
        log_gate.masked_fill_(~valid, float("-inf"))

    grid = lambda META: (
        num_seqs,
        num_heads,
        triton.cdiv(dim, META["BLOCK_D"]),
    )
    append_kv_cache_kernel[grid](
        k, v, log_gate, 
        k_cache, v_cache, log_gate_cache,
        write_pos,
        *(k.stride()),
        *(log_gate.stride()),
        *(k_cache.stride()),
        NUM_HEADS=num_heads,
        D=dim,
    )

    return valid

def init_buffer(
    num_groups: int,
    dim: int,
    buffer_size: int,
    device: torch.device = "cuda",
    dtype: torch.dtype = torch.float32,
):
    total_keys = num_groups * buffer_size

    k_cache = torch.empty(
        (total_keys, dim),
        device=device,
        dtype=dtype,
    )
    v_cache = torch.empty_like(k_cache)

    log_gate_cache = torch.empty(
        total_keys,
        device=device,
        dtype=dtype,
    )

    cu_seqlens_k = torch.arange(
        0,
        total_keys + 1,
        buffer_size,
        device=device,
        dtype=torch.int32,
    )

    write_pos = cu_seqlens_k[:-1].clone()

    return k_cache, v_cache, log_gate_cache, cu_seqlens_k, write_pos

def pad_buffer(
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    log_gate_cache: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    write_pos: torch.Tensor,
    buffer_size: int,
):
    """
    Args:
        k_cache, v_cache: (total_keys, dim)
        log_gate_cache: (total_keys,)
        cu_seqlens_k: (num_seqs * num_heads + 1,)
        write_pos: (num_seqs * num_heads,)
    
    Returns:
        k_out, v_out: (total_keys + num_groups * buffer_size, dim)
        log_gate_out: (total_keys + num_groups * buffer_size,)
        new_cu_seqlens_k: (num_seqs * num_heads + 1,)
        new_write_pos: (num_seqs * num_heads,)
    """
    num_groups = cu_seqlens_k.numel() - 1
    total_keys, dim = k_cache.shape

    real_lens = write_pos - cu_seqlens_k[:-1]

    if not torch.any(real_lens > 0):
        return k_cache, v_cache, log_gate_cache, cu_seqlens_k, write_pos


    avg_len = math.ceil(real_lens[real_lens > 0].float().mean().item())
    block_t = min(256, triton.next_power_of_2(avg_len))

    num_blocks = triton.cdiv(real_lens, block_t)

    pad_offsets = torch.zeros(
        num_groups + 1,
        dtype=torch.int32,
        device=k_cache.device,
    )
    pad_offsets[1:] = torch.cumsum(num_blocks, dim=0)

    total_blocks = pad_offsets[-1].item()

    new_group_lens = real_lens + buffer_size

    new_cu_seqlens_k = torch.empty_like(cu_seqlens_k)
    new_cu_seqlens_k[0] = 0
    new_cu_seqlens_k[1:] = torch.cumsum(new_group_lens, dim=0)

    new_total_keys = new_cu_seqlens_k[-1].item()
    new_write_pos = new_cu_seqlens_k[:-1] + real_lens

    k_out = torch.empty(
        (new_total_keys, dim),
        device=k_cache.device,
        dtype=k_cache.dtype,
    )
    v_out = torch.empty(
        (new_total_keys, dim),
        device=v_cache.device,
        dtype=v_cache.dtype,
    )
    log_gate_out = torch.empty(
        new_total_keys,
        device=log_gate_cache.device,
        dtype=log_gate_cache.dtype,
    )

    grid = lambda META: (
        total_blocks,
        triton.cdiv(dim, META["BLOCK_D"]),
    )
    pad_buffer_kernel[grid](
        k_cache, v_cache, log_gate_cache,
        k_out, v_out, log_gate_out,
        cu_seqlens_k, new_cu_seqlens_k, write_pos,
        pad_offsets,
        *(k_cache.stride()),
        *(k_out.stride()),
        NUM_GROUPS=num_groups,
        D=dim,
        BLOCK_T=block_t,
    )

    return k_out, v_out, log_gate_out, new_cu_seqlens_k, new_write_pos

def _qk_matmul(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    log_gate_cache: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    write_pos: torch.Tensor,
    group_lens: torch.Tensor,
    cu_seqlens_scores: torch.Tensor,
    avg_len: int,
    scale: float,
):
    """
    Args:
        q: (num_seqs, num_heads, dim)
        k_cache: (total_keys, dim)
        log_gate_cache: (total_keys,)
        cu_seqlens_k: (num_groups + 1,)
        write_pos: (num_groups,)

    Returns:
        scores: (total_keys,)
        cu_seqlens_scores: (num_groups + 1,)
    """
    num_seqs, num_heads, dim = q.shape
    num_groups = num_seqs * num_heads

    block_t = min(256, triton.next_power_of_2(int(avg_len)))

    num_token_blocks = triton.cdiv(group_lens, block_t)
    qk_offsets = torch.zeros(num_groups + 1, dtype=torch.int32, device=q.device)
    qk_offsets[1:] = torch.cumsum(num_token_blocks, dim=0)

    total_blocks = qk_offsets[-1].item()

    total_scores = cu_seqlens_scores[-1].item()

    scores = torch.empty(total_scores, device=q.device, dtype=q.dtype)

    qk_matmul_kernel[(total_blocks,)](
        q, k_cache, log_gate_cache,
        cu_seqlens_k, write_pos,
        scores, cu_seqlens_scores,
        qk_offsets,
        *(q.stride()),
        *(k_cache.stride()),
        NUM_GROUPS=num_groups,
        NUM_HEADS=num_heads,
        SCALE=scale,
        D=dim,
        BLOCK_T=block_t,
    )

    return scores

def _softmax(
    scores: torch.Tensor,
    cu_seqlens_scores: torch.Tensor,
    avg_len: int,
):
    """
    Args:
        scores: (total_scores,)
        cu_seqlens_scores: (num_groups + 1,)

    Returns:
        scores: (total_scores,)
    """
    num_groups = cu_seqlens_scores.numel() - 1

    block_t = min(256, triton.next_power_of_2(int(avg_len)))

    softmax_kernel[(num_groups,)](
        scores,
        cu_seqlens_scores,
        *scores.stride(),
        BLOCK_T=block_t,
    )

    return scores

def _attn_output(
    scores: torch.Tensor,
    v_cache: torch.Tensor,
    cu_seqlens_scores: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    avg_len: int,
    num_seqs: int,
    num_heads: int,
):
    """
    Args:
        scores: (total_keys,)
        v_cache: (total_keys, dim)
        cu_seqlens_scores, cu_seqlens_k: (num_groups + 1,)

    Returns:
        output: (num_seqs, num_heads, dim)
    """
    num_groups = num_seqs * num_heads
    _, dim = v_cache.shape

    block_t = min(256, triton.next_power_of_2(int(avg_len)))

    output = torch.empty(
        (num_seqs, num_heads, dim),
        device=v_cache.device, dtype=v_cache.dtype,
    )

    grid = lambda META: (
        num_groups,
        triton.cdiv(dim, META["BLOCK_D"]),
    )
    attn_v_kernel[grid](
        scores, v_cache,
        cu_seqlens_scores, cu_seqlens_k,
        output,
        *(v_cache.stride()),
        *(output.stride()),
        *(scores.stride()),
        NUM_HEADS=num_heads,
        D=dim,
        BLOCK_T=block_t,
    )

    return output

def varlen_min_self_attn_decode(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gate: torch.Tensor,
    scores_std: torch.Tensor,
    log_gate_penalty: float,
    scale: float,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    log_gate_cache: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    write_pos: torch.Tensor,
    gate_threshold: torch.Tensor | None = None
):
    """
    Args:
        q, k, v: (num_seqs, num_heads, dim)
        gate: (num_seqs, num_heads)
        scores_std: (num_heads,)
        k_cache, v_cache: (total_keys, dim)
        log_gate_cache: (total_keys,)
        cu_seqlens: (num_seqs + 1,)
        cu_seqlens_k: (num_groups + 1,)
        write_pos: (num_seqs * num_heads,)
        gate_threshold: (num_heads,)
    
    Returns:
        out: (num_seqs, num_heads, dim)
        k_cache, v_cache: (total_keys, dim)
        log_gate_cache: (total_keys,)
        write_pos: (num_seqs * num_heads,)
    """
    num_seqs, num_heads, dim = q.shape
    num_groups = num_seqs * num_heads

    valid = _append_kv_cache(
        k, v, gate, scores_std, log_gate_penalty,
        k_cache, v_cache, log_gate_cache,
        write_pos, gate_threshold
    )

    group_lens = write_pos - cu_seqlens_k[:-1] + 1
    cu_seqlens_scores = torch.zeros(num_groups + 1, dtype=torch.int32, device=q.device)
    cu_seqlens_scores[1:] = torch.cumsum(group_lens, dim=0)
    avg_len = math.ceil(group_lens.float().mean().item())

    scores = _qk_matmul(
        q, k_cache, log_gate_cache, 
        cu_seqlens_k, write_pos,
        group_lens, cu_seqlens_scores, avg_len, scale
    )

    _softmax(scores, cu_seqlens_scores, avg_len)
    
    out = _attn_output(
        scores, v_cache,
        cu_seqlens_scores, cu_seqlens_k,
        avg_len,
        num_seqs, num_heads
    )

    if gate_threshold is not None:
        write_pos += valid.reshape(-1).to(write_pos.dtype)
    else:
        write_pos += 1

    return out, k_cache, v_cache, log_gate_cache, write_pos
