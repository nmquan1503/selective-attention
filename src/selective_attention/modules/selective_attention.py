import torch
import torch.nn as nn
import torch.nn.functional as F

from ..inference import SelectiveAttnCache, InferenceState
from selective_attention.inference.generation_config import GenerationConfig

def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def build_rope_cache(seq_len_or_positions, dim, device, mode="seq"):
    """
    if mode == "seq":
        Returns: 
            cos, sin: (seq_len, dim)

    if mode == "pos":
        Returns:
            cos, sin: (batch_size, dim)
    """
    assert dim % 2 == 0

    half_dim = dim // 2
    freq = 1.0 / (10000 ** (torch.arange(0, half_dim, device=device).float() / half_dim))

    if mode == 'seq':
        seq_len = seq_len_or_positions
        pos = torch.arange(seq_len, device=device, dtype=torch.float32)
        angles = torch.einsum("i,j->ij", pos, freq)
    elif mode == 'pos':
        positions = seq_len_or_positions.float()
        angles = positions.unsqueeze(-1) * freq.unsqueeze(0) 
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    cos = torch.cos(angles).repeat_interleave(2, dim=-1)
    sin = torch.sin(angles).repeat_interleave(2, dim=-1)
    return cos, sin

def apply_rotary(q, k, cos, sin, mode="seq"):
    if mode == "seq":
        cos = cos[None, None, :, :]
        sin = sin[None, None, :, :]
    elif mode == "pos":
        cos = cos[:, None, :]
        sin = sin[:, None, :]
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    q_rot = q * cos + rotate_half(q) * sin
    k_rot = k * cos + rotate_half(k) * sin

    return q_rot, k_rot

def build_attn_matrix(attn_matrix, log_gate):
    """
    Args:
        attn_matrix: (batch_size, num_heads, seq_len, seq_len)
        log_gate: (batch_size, mlconv_radius + 1, seq_len)

    Returns:
        attn_matrix: (batch_size, num_heads, seq_len, seq_len)
    """
    batch_size, num_heads, seq_len, _ = attn_matrix.shape
    mlconv_radius = log_gate.shape[1] - 1
    device = attn_matrix.device

    idx = torch.arange(seq_len, device=device)
    rel = idx[:, None] - idx[None, :]
    future_mask = rel < 0
    dist = rel.clamp(min=0, max=mlconv_radius)
    level = mlconv_radius - dist
    log_gate = log_gate.unsqueeze(1)
    level_idx = level.unsqueeze(0).unsqueeze(0)
    level_idx = level_idx.expand(batch_size, num_heads, seq_len, seq_len)
    log_gate = torch.gather(
        log_gate.expand(batch_size, num_heads, mlconv_radius + 1, seq_len),
        dim=2,
        index=level_idx
    )
    attn_matrix = attn_matrix + log_gate
    attn_matrix = attn_matrix.masked_fill(future_mask.unsqueeze(0).unsqueeze(0), float("-inf"))
    return attn_matrix

class SelectiveMHA(nn.Module):
    def __init__(self, dim, head_dim):
        super().__init__()

        assert dim % head_dim == 0
        self.dim = dim
        self.num_heads = dim // head_dim
        self.head_dim = head_dim
        assert self.head_dim % 2 == 0, "head_dim must be even for RoPE"

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

    def forward(
        self, 
        hidden_states: torch.Tensor, 
        gate: torch.Tensor,
        lengths: torch.Tensor | None = None, 
        cache: SelectiveAttnCache | None = None
    ):
        """
        Args:
            hidden_states: (batch_size, seq_len, dim)
            lengths: (batch_size,)
            gate: (batch_size, mlconv_radius + 1, seq_len)
        
        Returns:
            hidden_states: (batch_size, seq_len, dim)
        """
        batch_size, seq_len, _ = hidden_states.shape
        is_infer = cache is not None
 
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        cos, sin = build_rope_cache(seq_len, self.head_dim, hidden_states.device)
        q_rot, k_rot = apply_rotary(q, k, cos, sin)

        scale = self.head_dim ** 0.5
        attn_matrix = (q_rot @ k_rot.transpose(-2, -1)) / scale
        log_gate = torch.log(gate.clamp(min=1e-12))
        attn_matrix = build_attn_matrix(attn_matrix, log_gate)
        
        attn_weights = F.softmax(attn_matrix, dim=-1)

        out = attn_weights @ v
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.dim)
        hidden_states = self.out_proj(out)
        
        if is_infer:
            cache.build(k_rot, v, log_gate, lengths)
        
        return hidden_states

    def step(
        self, 
        hidden_states: torch.Tensor, 
        gate: torch.Tensor, 
        cache: SelectiveAttnCache, 
        state: InferenceState,
        gen_cfg: GenerationConfig,
    ):
        """
        Args:
            hidden_states: (batch_size, model_dim)
            gate: (batch_size, mlconv_radius + 1)
        
        Returns:
            hidden_states: (batch_size, model_dim)
        """

        batch_size, _ = hidden_states.shape
        mlconv_radius = gate.shape[1] - 1
        device = hidden_states.device

        if state.step % gen_cfg.cache_update_interval == 0:
            cache.reset(mlconv_radius, gen_cfg.attn_gate_threshold, gen_cfg.cache_update_interval)
        
        log_gate = torch.log(gate)

        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        q = q.view(batch_size, self.num_heads, self.head_dim)
        k = k.view(batch_size, self.num_heads, self.head_dim)
        v = v.view(batch_size, self.num_heads, self.head_dim)

        cos, sin = build_rope_cache(cache.lengths, self.head_dim, device, mode='pos')
        q_rot, k_rot = apply_rotary(q, k, cos, sin, mode='pos')

        cache.update(k_rot, v, log_gate, gen_cfg.attn_gate_threshold)

        scale = self.head_dim ** 0.5
        attn_matrix = (q_rot.unsqueeze(2) @ cache.k_rot[:, :, :cache.write_idx, :].transpose(-2, -1)) / scale
        attn_matrix = attn_matrix + cache.log_gate[:, None, None, :cache.write_idx]

        attn_matrix = attn_matrix.masked_fill(
            cache.valid_mask[:, None, None, :cache.write_idx] == 0,
            float("-inf")
        )

        attn_matrix = F.softmax(attn_matrix, dim=-1)
        out = attn_matrix @ cache.v[:, :, :cache.write_idx, :]
        out = out.transpose(1, 2).contiguous().view(batch_size, self.dim)

        return self.out_proj(out)
