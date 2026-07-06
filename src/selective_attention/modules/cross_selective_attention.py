import torch
import torch.nn as nn

from ..inference import CrossSelectiveAttnCache, GenerationConfig, AnalysisConfig
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
        self.select_gate_proj = nn.Linear(dim, self.num_heads)
        # self.out_gate_proj = nn.Linear(dim, dim)

    def forward(
        self, 
        hidden_states: torch.Tensor, 
        context: torch.Tensor,
        context_lengths: torch.Tensor | None = None,
        attn_gate_threshold: float | None = None,
        cache: CrossSelectiveAttnCache | None = None,
        analysis_cfg: AnalysisConfig | None = None,
        stats: dict | None = None
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

        gate = torch.sigmoid(self.select_gate_proj(context)).transpose(1, 2).contiguous()
        # out_gate = torch.sigmoid(self.out_gate_proj(hidden_states))
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
        # hidden_states = self.out_proj(out * out_gate)
        hidden_states = self.out_proj(out)

        if analysis_cfg is not None and stats is not None:
            if context_lengths is not None:
                scale = context_lengths.view(batch_size, 1, 1, 1).to(dtype=attn_weights.dtype)
            else:
                scale = context_len
            scaled_weight = attn_weights * scale
            
            if context_lengths is not None:
                pos = torch.arange(context_len, device=device)
                key_valid = (pos.unsqueeze(0) < context_lengths.unsqueeze(1)).view(batch_size, 1, 1, context_len)
                scaled_weight = scaled_weight * key_valid

            sum_per_key = scaled_weight.sum(dim=2)
            count_per_key = (scaled_weight > 0).sum(dim=2)

            num_bins = analysis_cfg.gate_attn_num_bins
            bin_sum = torch.zeros((self.num_heads, num_bins), dtype=torch.float32, device=device)
            bin_count = torch.zeros((self.num_heads, num_bins), dtype=torch.float32, device=device)

            for b in range(batch_size):
                for h in range(self.num_heads):
                    for k in range(context_len):
                        cnt = count_per_key[b, h, k].item()
                        if cnt > 0:
                            g_val = gate[b, h, k].item()
                            if g_val == 0:
                                continue
                            s_val = sum_per_key[b, h, k].item()
                            bin_idx = max(0, min(int(g_val * num_bins), num_bins - 1))
                            bin_sum[h, bin_idx] += s_val
                            bin_count[h, bin_idx] += cnt

            stats["cross_attn_gate_analysis"] = {
                "sum": bin_sum,
                "count": bin_count
            }

        return hidden_states

    def step(
        self, 
        hidden_states: torch.Tensor, 
        cache: CrossSelectiveAttnCache,
        analysis_cfg: AnalysisConfig | None = None,
        stats: dict | None = None
    ):
        """
        Args:
            hidden_states: (batch_size, model_dim)
        
        Returns:
            hidden_states: (batch_size, model_dim)
        """
        batch_size = hidden_states.shape[0]
        device = hidden_states.device
        
        # out_gate = torch.sigmoid(self.out_gate_proj(hidden_states))
        q = self.q_proj(hidden_states)
        q = q.view(batch_size, self.num_heads, self.head_dim).unsqueeze(2)
        attn_matrix = (q @ cache.k.transpose(-2, -1)) * self.scale
        attn_weights = _gated_softmax(attn_matrix, cache.gate, is_infer=True)
        out = attn_weights @ cache.v
        out = out.squeeze(2).view(batch_size, self.dim)
        # hidden_states = self.out_proj(out * out_gate)
        hidden_states = self.out_proj(out)

        if analysis_cfg is not None and stats is not None:
            K = cache.gate.shape[2]
            scale = (cache.gate > 0).sum(dim=-1, keepdim=True).float()
            scaled_weight = attn_weights * scale.unsqueeze(2)
            key_valid_mask = (cache.gate > 0).unsqueeze(2)
            scaled_weight = scaled_weight * key_valid_mask
            sum_per_key = scaled_weight.sum(dim=2)
            count_per_key = (scaled_weight > 0).sum(dim=2)
            num_bins = analysis_cfg.gate_attn_num_bins
            key = "cross_attn_gate_analysis"
            bin_sum = stats[key]["sum"]
            bin_count = stats[key]["count"]
            for b in range(batch_size):
                for h in range(self.num_heads):
                    for k in range(K):
                        cnt = count_per_key[b, h, k].item()
                        if cnt > 0:
                            g_val = cache.gate[b, h, k].item()
                            if g_val == 0:
                                continue
                            s_val = sum_per_key[b, h, k].item()
                            bin_idx = max(0, min(int(g_val * num_bins), num_bins - 1))
                            bin_sum[h, bin_idx] += s_val
                            bin_count[h, bin_idx] += cnt

        return hidden_states