import torch
import torch.nn as nn

from ..inference import CrossSelectiveAttnCache, GenerationConfig
from ..utils.tensor_utils import compress

def _gated_softmax(attn_matrix, gate, is_infer, eps=1e-12):
    """
    Args:
        attn_matrix: (batch_size, num_heads, seq_len, num_keys)
        gate: (batch_size, num_heads, num_keys)
    """
    max_val = attn_matrix.max(dim=-1, keepdim=True).values
    if is_infer:
        attn_matrix.sub_(max_val)
        attn_matrix.exp_()
        exp_attn = attn_matrix
        exp_attn[:, :, :, 1:].mul_(gate[:, :, 1:].unsqueeze(2))
        numerator = exp_attn
    else:
        exp_attn = torch.exp(attn_matrix - max_val)
        numerator = exp_attn * gate.unsqueeze(2)
        numerator[:, :, :, 0] = exp_attn[:, :, :, 0]
    denom = numerator.sum(dim=-1, keepdim=True)

    if is_infer:
        numerator.div_(denom + eps)
        attn_weight = numerator
    else:
        attn_weight = numerator / (denom + eps)
    
    return attn_weight

class CrossSelectiveMHA(nn.Module):
    def __init__(
        self, 
        layer_idx: int,
        dim: int, 
        head_dim: int
    ):
        super().__init__()
        self.layer_idx = layer_idx

        assert dim % head_dim == 0
        self.dim = dim
        self.num_heads = dim // head_dim
        self.head_dim = head_dim
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.gate_proj = nn.Linear(dim, self.num_heads)

    def forward(
        self, 
        hidden_states: torch.Tensor, 
        context: torch.Tensor,
        context_lengths: torch.Tensor | None = None,
        attn_gate_threshold: float | None = None,
        cache: CrossSelectiveAttnCache | None = None
    ):
        """
        Args:
            hidden_states: (batch_size, seq_len, model_dim)
            context: (batch_size, context_len, model_dim)
            context_lengths: (batch_size,)

        Returns:
            hidden_states: (batch_size, seq_len, model_dim)
        """
        batch_size, seq_len, _ = hidden_states.shape
        context_len = context.shape[1]
        device = hidden_states.device
        is_infer = not self.training
        is_prefill = is_infer and cache is not None

        gate = torch.sigmoid(self.gate_proj(context)).transpose(1, 2).contiguous()
        if context_lengths is not None:
            valid_mask = torch.arange(context_len, device=device)[None, :] < context_lengths[:, None]
            if not is_infer:
                gate = gate * valid_mask[:, None, :]
            else:
                gate *= valid_mask[:, None, :]

        q = self.q_proj(hidden_states)
        k = self.k_proj(context)
        v = self.v_proj(context)
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        v = v.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

        if is_prefill:
            if attn_gate_threshold is not None:
                select_mask = (gate >= attn_gate_threshold) & (gate > 0.0)
                select_mask[:, :, 0] = True
                gate = compress(gate.unsqueeze(-1), select_mask).squeeze(-1)
                k = compress(k, select_mask)
                v = compress(v, select_mask)

            cache.k = k
            cache.v = v
            cache.gate = gate
        
        attn_matrix = (q @ k.transpose(-2, -1)) * self.scale
        attn_weights = _gated_softmax(attn_matrix, gate, is_infer=is_infer)
        out = attn_weights @ v
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.dim)
        hidden_states = self.out_proj(out)

        return hidden_states

    def step(self, hidden_states: torch.Tensor, cache: CrossSelectiveAttnCache):
        """
        Args:
            hidden_states: (batch_size, model_dim)
        
        Returns:
            hidden_states: (batch_size, model_dim)
        """
        batch_size = hidden_states.shape[0]

        q = self.q_proj(hidden_states)
        q = q.view(batch_size, self.num_heads, self.head_dim).unsqueeze(2)
        attn_matrix = (q @ cache.k.transpose(-2, -1)) * self.scale
        attn_weights = _gated_softmax(attn_matrix, cache.gate, is_infer=True)
        out = attn_weights @ cache.v
        out = out.squeeze(2).view(batch_size, self.dim)
        hidden_states = self.out_proj(out)
        return hidden_states