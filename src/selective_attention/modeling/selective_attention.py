import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from selective_attention.modeling.inference_state import InferenceState
from selective_attention.modeling.generation_config import GenerationConfig

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
        cos = cos[:, None, None, :]
        sin = sin[:, None, None, :]
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

def right_align(
    x: torch.Tensor,
    valid_mask: torch.Tensor,
    buffer_size: int
):
    """
    Args:
        x: (batch_size, num_heads, seq_len, head_dim)
        valid_mask: (batch_size, seq_len)
    
    Returns:
        x: (batch_size, num_heads, compressed_len + buffer_size, head_dim)
    """
    batch_size, num_heads, seq_len, head_dim = x.shape
    device = x.device

    num_valid = valid_mask.sum(dim=1)
    max_valid = num_valid.max().item()
    rank = valid_mask.cumsum(dim=1) - 1
    shift = max_valid - num_valid
    out = torch.empty(
        batch_size, num_heads, max_valid + buffer_size, head_dim,
        device=device, dtype=torch.float32
    )
    src_valid = x.permute(0, 2, 1, 3)[valid_mask]
    batch_idx = torch.arange(batch_size, device=device)
    batch_idx = batch_idx.repeat_interleave(num_valid)
    dst_valid = rank[valid_mask] + shift.repeat_interleave(num_valid)
    out[batch_idx, :, dst_valid, :] = src_valid
    valid_mask = (
        torch.arange(max_valid + buffer_size, device=device)
        .unsqueeze(0) 
        >= shift.unsqueeze(1)
    ) & (
        torch.arange(max_valid + buffer_size, device=device)
        .unsqueeze(0)
        < (shift + num_valid).unsqueeze(1)
    )
    return out, valid_mask

class SelectiveMHA(nn.Module):
    def __init__(self, layer_idx, dim, head_dim):
        super().__init__()
        self.layer_idx = layer_idx

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
        state: InferenceState | None = None
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
        mlconv_radius = gate.shape[1] - 1
        device = hidden_states.device
        is_infer = state is not None
        if is_infer:
            layer_state = state.layers[self.layer_idx]

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
            layer_state.k_rot = k_rot
            layer_state.v = v
            layer_state.lengths = lengths
            
            pos = torch.arange(seq_len, device=device).unsqueeze(0)
            level_idx = torch.clamp(
                pos - (lengths.unsqueeze(1) - 1 - mlconv_radius),
                min=0,
                max=mlconv_radius
            )
            layer_state.log_gate = log_gate[
                torch.arange(batch_size, device=device).unsqueeze(1),
                level_idx,
                pos
            ]
        
        return hidden_states

    def step(
        self, 
        hidden_states: torch.Tensor, 
        gate: torch.Tensor, 
        state: InferenceState, 
        gen_cfg: GenerationConfig
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
        layer_state = state.layers[self.layer_idx]

        if state.step % gen_cfg.cache_update_interval == 0:
            if state.step == 0:
                compressed_len = layer_state.v.shape[2]
                pos = torch.arange(compressed_len, device=device).unsqueeze(0)
                valid_mask = pos < layer_state.lengths.unsqueeze(1)
                gate_mask = layer_state.log_gate >= math.log(gen_cfg.attn_gate_threshold)
                tail_start = layer_state.lengths - mlconv_radius
                tail_mask = pos >= tail_start.unsqueeze(1)
                layer_state.valid_mask = valid_mask & (gate_mask | tail_mask)
            layer_state.log_gate, _ = right_align(
                layer_state.log_gate.unsqueeze(1).unsqueeze(-1),
                layer_state.valid_mask,
                buffer_size=gen_cfg.cache_update_interval
            )
            layer_state.log_gate = layer_state.log_gate.squeeze(-1).squeeze(1)
            layer_state.k_rot, _ = right_align(
                layer_state.k_rot,
                layer_state.valid_mask,
                buffer_size=gen_cfg.cache_update_interval
            )
            layer_state.v, layer_state.valid_mask = right_align(
                layer_state.v,
                layer_state.valid_mask,
                buffer_size=gen_cfg.cache_update_interval
            )
            layer_state.write_idx = layer_state.v.shape[2] - gen_cfg.cache_update_interval

        log_gate = torch.log(gate)

        write_idx = layer_state.write_idx
        remove_mask = gate[:, 0] < gen_cfg.attn_gate_threshold
        layer_state.valid_mask[remove_mask, write_idx-mlconv_radius] = False
        layer_state.valid_mask[:, write_idx] = True

        layer_state.log_gate[:, write_idx-mlconv_radius : write_idx+1] = log_gate

        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        q = q.view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)

        cos, sin = build_rope_cache(layer_state.lengths, self.head_dim, device, mode='pos')
        q_rot, k_rot = apply_rotary(q, k, cos, sin, mode='pos')

        layer_state.k_rot[:, :, write_idx:write_idx+1, :] = k_rot
        layer_state.v[:, :, write_idx:write_idx+1, :] = v

        layer_state.lengths += 1
        layer_state.write_idx += 1
 
        scale = self.head_dim ** 0.5
        attn_matrix = (q_rot @ layer_state.k_rot[:, :, :write_idx+1, :].transpose(-2, -1)) / scale
        attn_matrix = attn_matrix + layer_state.log_gate[:, None, None, :write_idx+1]

        attn_matrix = attn_matrix.masked_fill(
            layer_state.valid_mask[:, None, None, :write_idx+1] == 0,
            float("-inf")
        )

        attn_matrix = F.softmax(attn_matrix, dim=-1)
        out = attn_matrix @ layer_state.v[:, :, :write_idx+1, :]
        out = out.transpose(1, 2).contiguous().view(batch_size, self.dim)

        return self.out_proj(out)
