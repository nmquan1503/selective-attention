import torch
import torch.nn as nn
import torch.nn.functional as F

from ..inference import SelectiveAttnCache, InferenceState, GenerationConfig
from .rope import RoPE

def _build_attn_matrix(
    q: torch.Tensor,
    k: torch.Tensor,
):
    """
    Args:
        q, k: (batch_size, num_heads, seq_len, head_dim)
    
    Returns:
        attn_matrix: (batch_size, num_heads, seq_len, seq_len)
    """
    batch_size, _, seq_len, head_dim = q.shape
    device = q.device

    attn_matrix = torch.matmul(q, k.transpose(-2, -1)) * (head_dim ** -0.5)
    causal_mask = torch.triu(
        torch.ones(seq_len, seq_len, dtype=torch.bool, device=device),
        diagonal=1
    )
    attn_matrix = attn_matrix.masked_fill(causal_mask, float("-inf"))

    return attn_matrix

def _gated_softmax(
    attn_matrix: torch.Tensor,
    gate: torch.Tensor,
    eps: float = 1e-12
):
    """
    Args:
        attn_matrix: (batch_size, num_heads, seq_len, seq_len)
        gate: (batch_size, seq_len)
    """
    gate = gate[:, None, None, :]
    max_val = attn_matrix.max(dim=-1, keepdim=True).values
    exp_attn = torch.exp(attn_matrix - max_val)
    numerator = exp_attn * gate
    diag_indices = torch.arange(attn_matrix.size(-1), device=attn_matrix.device)
    numerator[..., diag_indices, diag_indices] = exp_attn[..., diag_indices, diag_indices]
    denom = numerator.sum(dim=-1, keepdim=True)
    attn_weight = numerator / (denom + eps)
    return attn_weight

class SelectiveMHA(nn.Module):
    def __init__(self, dim, head_dim, conv_kernel_size):
        super().__init__()

        assert dim % head_dim == 0
        self.dim = dim
        self.num_heads = dim // head_dim
        self.head_dim = head_dim
        self.conv_kernel_size = conv_kernel_size

        self.causal_conv = nn.Conv1d(
            in_channels=dim,
            out_channels=dim,
            kernel_size=conv_kernel_size,
            groups=self.num_heads,
            padding=conv_kernel_size - 1,
            bias=False
        )
        self.gate_proj = nn.Linear(dim, dim + 1)
        self.rope = RoPE(self.head_dim)
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

        nn.init.constant_(self.gate_proj.bias, 2.0)

    def forward(
        self, 
        hidden_states: torch.Tensor, 
        lengths: torch.Tensor | None = None,
        attn_gate_threshold: float = 0.5,
        cache: SelectiveAttnCache | None = None
    ):
        """
        Args:
            hidden_states: (batch_size, seq_len, dim)
            lengths: (batch_size,)
        
        Returns:
            hidden_states: (batch_size, seq_len, dim)
        """
        batch_size, seq_len, _ = hidden_states.shape
        device = hidden_states.device
        is_infer = cache is not None
        is_causal = True

        gate = torch.sigmoid(self.gate_proj(hidden_states))
        select_gate, out_gate = torch.split(gate, [1, self.dim], dim=-1)
        select_gate = select_gate.squeeze(-1)
        
        hidden_states = hidden_states.transpose(1, 2)
        if is_infer:
            cache.build_conv_ctx(hidden_states, lengths, self.conv_kernel_size)
        hidden_states = self.causal_conv(hidden_states)
        hidden_states = hidden_states.transpose(1, 2)
        hidden_states = hidden_states[:, :seq_len]

        hard_select_gate = (select_gate > attn_gate_threshold).float()
        if not is_infer:
            hard_select_gate = hard_select_gate.detach() + select_gate - select_gate.detach()
 
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        positions = torch.arange(seq_len, device=device)
        q_rot, k_rot = self.rope(q, k, positions, mode="seq")

        attn_matrix = _build_attn_matrix(q_rot, k_rot)
        attn_weight = _gated_softmax(attn_matrix, hard_select_gate)

        out = attn_weight @ v
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.dim)
        hidden_states = self.out_proj(out * out_gate)
        
        if is_infer and is_causal:
            cache.build_kv(k_rot, v, hard_select_gate, lengths)
        
        return hidden_states

    def step(
        self, 
        hidden_states: torch.Tensor, 
        cache: SelectiveAttnCache, 
        state: InferenceState,
        gen_cfg: GenerationConfig,
    ):
        """
        Args:
            hidden_states: (batch_size, model_dim)
        
        Returns:
            hidden_states: (batch_size, model_dim)
        """

        batch_size, _ = hidden_states.shape
        device = hidden_states.device

        if state.step % gen_cfg.cache_update_interval == 0:
            cache.reset(gen_cfg.cache_update_interval)
        
        gate = torch.sigmoid(self.gate_proj(hidden_states))
        select_gate, out_gate = torch.split(gate, [1, self.dim], dim=-1)
        select_gate = select_gate.squeeze(-1)
        hard_select_gate = (select_gate > gen_cfg.attn_gate_threshold).float()
        
        cache.update_conv_ctx(hidden_states)
        hidden_states = cache.conv_ctx
        hidden_states = F.conv1d(
            hidden_states, 
            weight=self.causal_conv.weight, 
            bias=self.causal_conv.bias,
            groups=self.causal_conv.groups
        ).squeeze(-1)

        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        q = q.view(batch_size, self.num_heads, self.head_dim)
        k = k.view(batch_size, self.num_heads, self.head_dim)
        v = v.view(batch_size, self.num_heads, self.head_dim)

        q_rot, k_rot = self.rope(q, k, state.lengths, mode="pos")

        cache.update_kv(k_rot, v, hard_select_gate)

        scale = self.head_dim ** 0.5
        attn_matrix = (q_rot.unsqueeze(2) @ cache.k_rot[:, :, :cache.write_idx, :].transpose(-2, -1)) / scale
        valid_mask = cache.valid_mask[:, :cache.write_idx]
        cur_valid = valid_mask[:, -1]
        valid_mask[:, -1] = True
        attn_matrix = attn_matrix.masked_fill(
            valid_mask[:, None, None, :] == 0,
            float("-inf")
        )
        valid_mask[:, -1] = cur_valid

        attn_weight = F.softmax(attn_matrix, dim=-1)
        out = attn_weight @ cache.v[:, :, :cache.write_idx, :]
        out = out.transpose(1, 2).contiguous().view(batch_size, self.dim)

        return self.out_proj(out * out_gate)
