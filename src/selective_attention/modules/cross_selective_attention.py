import torch
import torch.nn as nn

from ..inference import CrossSelectiveAttnCache
from ..utils.validation import check_required

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
        exp_attn.mul_(gate.unsqueeze(2))
        numerator = exp_attn
    else:
        exp_attn = torch.exp(attn_matrix - max_val)
        numerator = exp_attn * gate.unsqueeze(2)
    denom = numerator.sum(dim=-1, keepdim=True)

    if is_infer:
        numerator.div_(denom + eps)
        attn_weight = numerator
    else:
        attn_weight = numerator / (denom + eps)
    
    return attn_weight

def _compress(
    x: torch.Tensor,
    valid_mask: torch.Tensor,
    return_new_mask: bool = False
):
    """    
    Args:
        x: (batch_size, num_heads, seq_len, head_dim)
        valid_mask: (batch_size, num_heads, seq_len)
    
    Returns:
        out: (batch_size, num_heads, max_valid, head_dim)
        new_valid:  (batch_size, num_heads, max_valid)
    """
    batch_size, num_heads, seq_len, head_dim = x.shape
    device = x.device

    num_valid = valid_mask.sum(dim=-1)
    max_valid = num_valid.max().item()

    rank = valid_mask.cumsum(dim=-1) - 1
    new_len = max_valid

    out = torch.zeros(batch_size, num_heads, new_len, head_dim, device=device, dtype=x.dtype)

    flat_bh = batch_size * num_heads
    out_flat = out.view(flat_bh, new_len, head_dim)
    x_flat = x.view(flat_bh, seq_len, head_dim)
    valid_flat = valid_mask.view(flat_bh, seq_len)
    rank_flat = rank.view(flat_bh, seq_len)

    sel_idx = rank_flat[valid_flat]
    bh_idx = torch.arange(flat_bh, device=device).unsqueeze(-1).expand(-1, seq_len)
    bh_idx_sel = bh_idx[valid_flat]

    out_flat[bh_idx_sel, sel_idx] = x_flat[valid_flat]

    if not return_new_mask:
        return out

    num_valid_flat = num_valid.view(flat_bh, 1)
    pos_idx = torch.arange(new_len, device=device).view(1, -1)
    new_valid_flat = pos_idx < num_valid_flat
    new_valid = new_valid_flat.view(batch_size, num_heads, new_len)

    return out, new_valid

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

    def forward(
        self, 
        hidden_states: torch.Tensor, 
        context: torch.Tensor | None = None,
        context_valid_mask: torch.Tensor | None = None,
        gate: torch.Tensor | None = None,
        attn_gate_threshold: float | None = None,
        cache: CrossSelectiveAttnCache | None = None
    ):
        """
        Args:
            hidden_states: (batch_size, seq_len, model_dim)
            context: (batch_size, context_len, model_dim)
            context_valid_mask: (batch_size, num_heads, context_len)
            gate: (batch_size, num_heads, context_len)

        Returns:
            hidden_states: (batch_size, seq_len, model_dim)
        """
        batch_size, seq_len, _ = hidden_states.shape
        device = hidden_states.device
        is_infer = not self.training
        check_required(cache, "cache", is_infer, "inference")
        is_prefill = is_infer and cache.k is None
        check_required(context, "context", self.training, "training")
        check_required(context_valid_mask, "context_valid_mask", self.training, "training")
        check_required(gate, "gate", self.training, "training")
        check_required(context, "context", is_prefill, "prefill")
        check_required(context_valid_mask, "context_valid_mask", is_prefill, "prefill")
        check_required(gate, "gate", is_prefill, "prefill")

        if attn_gate_threshold is None:
            attn_gate_threshold = 0.0

        q = self.q_proj(hidden_states)
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        if self.training or is_prefill:
            k = self.k_proj(context)
            v = self.v_proj(context)
            k = k.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
            v = v.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
            if is_prefill:
                select_mask = gate >= attn_gate_threshold
                context_valid_mask &= select_mask
                
                gate = _compress(gate.unsqueeze(-1), context_valid_mask).squeeze(-1)
                k = _compress(k, context_valid_mask)
                v, context_valid_mask = _compress(v, context_valid_mask, return_new_mask=True)

                cache.k = k
                cache.v = v
                cache.gate = gate
                cache.valid_mask = context_valid_mask

        else:
            k = cache.k
            v = cache.v
            gate = cache.gate
            context_valid_mask = cache.valid_mask
        
        attn_matrix = (q @ k.transpose(-2, -1)) * self.scale
        if self.training:
            attn_matrix = attn_matrix.masked_fill(~context_valid_mask[:, :, None, :], float("-inf"))
        else:
            attn_matrix.masked_fill_(context_valid_mask[:, :, None, :], float("-inf"))
        
        attn_weights = _gated_softmax(attn_matrix, gate, is_infer=is_infer)
        out = attn_weights @ v
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.dim)
        hidden_states = self.out_proj(out)

        return hidden_states