import torch
import triton
import math

from .forward_kernel import (
    pad_buffer_kernel,
    min_attn_decode_kernel,
    reduce_kernel
)

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

    group_lens = write_pos - cu_seqlens_k[:-1] + 1
    max_len = group_lens.max().item() + 1
    avg_len = math.ceil(max_len / num_groups)

    CHUNK_SIZE_MIN = 16
    num_sms = torch.cuda.get_device_properties(
        torch.cuda.current_device()
    ).multi_processor_count
    chunk_size = max(
        CHUNK_SIZE_MIN,
        math.ceil(max_len / max(1, math.ceil(num_sms / num_groups))),
    )
    chunk_size = 1 << (chunk_size - 1).bit_length()
    num_chunks = math.ceil(max_len / chunk_size)

    MIN_BLOCK_K = 16
    MAX_BLOCK_K = 128
    dtype_size = q.element_size()
    SMEM_BUDGET = 64 * 1024

    # Peak SRAM ≈ (BLOCK_K * D + 2 * D + 2 * BLOCK_K) * dtype_size
    max_block_k = (
        SMEM_BUDGET // dtype_size - 2 * dim
    ) // (dim + 2)

    max_block_k = max(1, max_block_k)
    max_block_k = 1 << (max_block_k.bit_length() - 1)

    BLOCK_K = min(
        MAX_BLOCK_K,
        chunk_size,
        max_block_k,
    )
    BLOCK_K = max(MIN_BLOCK_K, BLOCK_K)

    mid_o = torch.zeros(num_groups, num_chunks, dim, dtype=q.dtype, device=q.device)
    mid_logsumexp = torch.zeros(num_groups, num_chunks, dtype=q.dtype, device=q.device)

    min_attn_decode_kernel[(num_seqs, num_heads, num_chunks)](
        q, k, v, gate, scores_std,
        gate_threshold,
        k_cache, v_cache, log_gate_cache,
        cu_seqlens_k, write_pos,
        mid_o, mid_logsumexp,
        *(q.stride()),
        *(gate.stride()),
        *(k_cache.stride()),
        *(mid_o.stride()),
        *(mid_logsumexp.stride()),
        NUM_HEADS=num_heads,
        SCALE=scale,
        D=dim,
        HAS_GATE_THRESHOLD=gate_threshold is not None,
        LOG_GATE_PENALTY=log_gate_penalty,
        CHUNK_SIZE=chunk_size,
        BLOCK_K=BLOCK_K
    )

    out = torch.empty_like(v)
    reduce_kernel[(num_seqs, num_heads)](
        gate, gate_threshold,
        mid_o, mid_logsumexp,
        cu_seqlens_k, write_pos,
        out,
        *(gate.stride()),
        *(mid_o.stride()),
        *(mid_logsumexp.stride()),
        *(out.stride()),
        NUM_HEADS=num_heads,
        D=dim,
        HAS_GATE_THRESHOLD=gate_threshold is not None,
        CHUNK_SIZE=chunk_size
    )
    
    return out, k_cache, v_cache, log_gate_cache, write_pos
