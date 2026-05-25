import torch
import torch.nn as nn
import torch.nn.functional as F

from ..inference import SelectiveAttnCache, InferenceState, GenerationConfig
from .rope import RoPE

def _build_gate_matrix(
    hard_gate: torch.Tensor, 
    lengths: torch.Tensor,
    num_heads: int,
):
    """
    Args:
        attn_matrix: (batch_size, num_heads, seq_len, seq_len)
        hard_gate: (batch_size, mlconv_radius + 1, seq_len) | (batch_size, seq_len)

    Returns:
        attn_matrix: (batch_size, num_heads, seq_len, seq_len)
    """
    batch_size = hard_gate.shape[0]
    seq_len = hard_gate.shape[-1]
    device = hard_gate.device
    is_causal = hard_gate.ndim == 3

    idx = torch.arange(seq_len, device=device)
    rel = idx[:, None] - idx[None, :]
    diag_mask = (rel == 0)
    
    if is_causal:
        mlconv_radius = hard_gate.shape[1] - 1
        dist = rel.clamp(min=0, max=mlconv_radius)
        level = mlconv_radius - dist
        hard_gate = hard_gate.unsqueeze(1)
        level_idx = level.unsqueeze(0).unsqueeze(0)
        level_idx = level_idx.expand(batch_size, num_heads, seq_len, seq_len)
        hard_gate_matrix = torch.gather(
            hard_gate.expand(batch_size, num_heads, mlconv_radius + 1, seq_len),
            dim=2,
            index=level_idx
        )
        hard_gate_matrix = hard_gate_matrix.masked_fill(
            diag_mask.unsqueeze(0).unsqueeze(0),
            1.0
        )
        future_mask = rel < 0
        valid_mask = ~future_mask
        valid_mask = valid_mask.unsqueeze(0).unsqueeze(0)

    else:
        hard_gate_matrix = hard_gate[:, None, None, :].expand(
            batch_size, num_heads, seq_len, seq_len
        )
        hard_gate_matrix = hard_gate_matrix.masked_fill(
            diag_mask.unsqueeze(0).unsqueeze(0),
            1.0
        )

        pad_mask = idx[None, :] < lengths[:, None]
        valid_mask = pad_mask[:, None, None, :]
        valid_mask = valid_mask.expand(batch_size, num_heads, seq_len, seq_len)

    return hard_gate_matrix, valid_mask

def _gated_softmax(
    attn_matrix: torch.Tensor, 
    gate: torch.Tensor, 
    valid_mask: torch.Tensor | None = None,
    eps: float = 1e-12
):
    if valid_mask is not None:
        attn_matrix = attn_matrix.masked_fill(~valid_mask, float("-inf"))
    max_val = attn_matrix.max(dim=-1, keepdim=True).values
    exp_attn_matrix = torch.exp(attn_matrix - max_val)
    numerator = exp_attn_matrix * gate
    denom = numerator.sum(dim=-1, keepdim=True)
    attn_weight = numerator / (denom + eps)
    return attn_weight

class SelectiveMHA(nn.Module):
    def __init__(self, dim, head_dim):
        super().__init__()

        assert dim % head_dim == 0
        self.dim = dim
        self.num_heads = dim // head_dim
        self.head_dim = head_dim
        assert self.head_dim % 2 == 0, "head_dim must be even for RoPE"

        self.rope = RoPE(self.head_dim)
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

    def forward(
        self, 
        hidden_states: torch.Tensor, 
        gate: torch.Tensor,
        lengths: torch.Tensor | None = None, 
        gate_threshold: float | None = None,
        cache: SelectiveAttnCache | None = None
    ):
        """
        Args:
            hidden_states: (batch_size, seq_len, dim)
            lengths: (batch_size,)
            gate: (batch_size, mlconv_radius + 1, seq_len) | (batch_size, seq_len)
        
        Returns:
            hidden_states: (batch_size, seq_len, dim)
        """
        batch_size, seq_len, _ = hidden_states.shape
        device = hidden_states.device
        is_infer = cache is not None
        is_causal = gate.ndim == 3

        if is_infer:
            hard_gate = (gate > gate_threshold).float()
        else:
            hard_gate = (gate > 0.5).float()
            hard_gate = hard_gate.detach() + gate - gate.detach()
 
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        positions = torch.arange(seq_len, device=device)
        q_rot, k_rot = self.rope(q, k, positions, mode="seq")

        scale = self.head_dim ** 0.5
        attn_matrix = (q_rot @ k_rot.transpose(-2, -1)) / scale
        hard_gate_matrix, valid_mask = _build_gate_matrix(
            hard_gate=hard_gate,
            lengths=lengths,
            num_heads=self.num_heads,
        )
        attn_weight = _gated_softmax(attn_matrix, hard_gate_matrix, valid_mask)

        out = attn_weight @ v
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.dim)
        hidden_states = self.out_proj(out)
        
        if is_infer and is_causal:
            cache.build(k_rot, v, hard_gate, lengths)
        
        if not is_infer:
            return hidden_states, hard_gate_matrix, attn_weight, valid_mask

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
            cache.reset(mlconv_radius, gen_cfg.cache_update_interval)
        
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        q = q.view(batch_size, self.num_heads, self.head_dim)
        k = k.view(batch_size, self.num_heads, self.head_dim)
        v = v.view(batch_size, self.num_heads, self.head_dim)

        q_rot, k_rot = self.rope(q, k, state.lengths, mode="pos")

        cache.update(k_rot, v, gate > gen_cfg.attn_gate_threshold)

        scale = self.head_dim ** 0.5
        attn_matrix = (q_rot.unsqueeze(2) @ cache.k_rot[:, :, :cache.write_idx, :].transpose(-2, -1)) / scale
        valid_mask = cache.valid_mask[:, :cache.write_idx].clone()
        valid_mask[:, -1] = True
        attn_matrix = attn_matrix.masked_fill(
            valid_mask[:, None, None, :] == 0,
            float("-inf")
        )

        attn_weight = F.softmax(attn_matrix, dim=-1)
        out = attn_weight @ cache.v[:, :, :cache.write_idx, :]
        out = out.transpose(1, 2).contiguous().view(batch_size, self.dim)

        return self.out_proj(out)
