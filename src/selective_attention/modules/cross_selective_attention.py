import torch
import torch.nn as nn
import torch.nn.functional as F

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

        nn.init.constant_(self.select_gate_proj.bias, 2.0)

    def forward(
        self, 
        hidden_states: torch.Tensor, 
        context: torch.Tensor,
        context_lengths: torch.Tensor | None = None,
        attn_gate_threshold: torch.Tensor | None = None,
        cache: CrossSelectiveAttnCache | None = None,
        analysis_cfg: AnalysisConfig | None = None,
        stats: dict | None = None
    ):
        """
        Args:
            hidden_states: (batch_size, seq_len, model_dim)
            context: (batch_size, context_len, model_dim)
            context_lengths: (batch_size,)
            attn_gate_threshold: (num_heads,)

        Returns:
            hidden_states: (batch_size, seq_len, model_dim)
        """
        batch_size, seq_len, _ = hidden_states.shape
        context_len = context.shape[1]
        device = hidden_states.device
        is_infer = not self.training
        is_prefill = is_infer and cache is not None

        select_gate = torch.sigmoid(self.select_gate_proj(context)).transpose(1, 2).contiguous()

        q = self.q_proj(hidden_states)
        k = self.k_proj(context)
        v = self.v_proj(context)
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        v = v.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

        if is_prefill:
            select_gate[:, :, 0] = 1
            if context_lengths is not None:
                valid_mask = torch.arange(context_len, device=device)[None, :] < context_lengths[:, None]
                if not is_infer:
                    select_gate = select_gate * valid_mask[:, None, :]
                else:
                    select_gate *= valid_mask[:, None, :]
            if attn_gate_threshold is not None:
                select_mask = (select_gate >= attn_gate_threshold[None, :, None]) & (select_gate > 0.0)
                select_mask[:, :, 0] = True
                select_gate = compress(select_gate.unsqueeze(-1), select_mask).squeeze(-1)
                k = compress(k, select_mask)
                v = compress(v, select_mask)

            log_select_gate = torch.log(select_gate)
            cache.k = k
            cache.v = v
            cache.log_gate = log_select_gate
        else:
            log_select_gate = torch.log(select_gate)
            if context_lengths is not None:
                valid_mask = torch.arange(context_len, device=device)[None, :] < context_lengths[:, None]
                log_select_gate = log_select_gate.masked_fill(
                    ~valid_mask[:, None, :],
                    float("-inf"),
                )
        
        attn_matrix = (q @ k.transpose(-2, -1)) * self.scale
        if is_infer:
            attn_matrix[:, :, :, 1:] += log_select_gate[:, :, 1:].unsqueeze(2)
        else:
            attn_matrix[:, :, :, 1:] = attn_matrix[:, :, :, 1:] + log_select_gate[:, :, 1:].unsqueeze(2)
        attn_weights = F.softmax(attn_matrix, dim=-1)
        out = attn_weights @ v
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.dim)
        hidden_states = self.out_proj(out)

        if analysis_cfg is not None and stats is not None and attn_gate_threshold is None:
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
            bin_idx = (select_gate * num_bins).long().clamp(0, num_bins - 1)
            head_idx = torch.arange(self.num_heads, device=device).view(1, self.num_heads, 1).expand(batch_size, -1, context_len).reshape(-1)
            flat_index = head_idx * num_bins + bin_idx.reshape(-1)

            bin_attn_mass_flat = torch.zeros(self.num_heads * num_bins, dtype=torch.float32, device=device)
            bin_attn_count_flat = torch.zeros_like(bin_attn_mass_flat)
            bin_gate_freq_flat = torch.zeros_like(bin_attn_mass_flat)

            bin_attn_mass_flat.scatter_add_(0, flat_index, sum_per_key.reshape(-1))
            bin_attn_count_flat.scatter_add_(0, flat_index, count_per_key.reshape(-1).float())
            gate_ones = torch.ones_like(flat_index, dtype=torch.float32)
            bin_gate_freq_flat.scatter_add_(0, flat_index, gate_ones)

            stats["cross_attn_gate_analysis"] = {
                "attn_mass": bin_attn_mass_flat.view(self.num_heads, num_bins),
                "attn_count": bin_attn_count_flat.view(self.num_heads, num_bins),
                "gate_freq": bin_gate_freq_flat.view(self.num_heads, num_bins)
            }

        return hidden_states

    def step(
        self, 
        hidden_states: torch.Tensor, 
        cache: CrossSelectiveAttnCache,
        gen_cfg: GenerationConfig,
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
        attn_gate_threshold = gen_cfg.attn_gate_thresholds[self.layer_idx] if gen_cfg.attn_gate_thresholds is not None else None
        
        q = self.q_proj(hidden_states)
        q = q.view(batch_size, self.num_heads, self.head_dim).unsqueeze(2)
        attn_matrix = (q @ cache.k.transpose(-2, -1)) * self.scale
        attn_matrix += cache.log_gate.unsqueeze(2)
        attn_weights = F.softmax(attn_matrix, dim=-1)
        out = attn_weights @ cache.v
        out = out.squeeze(2).view(batch_size, self.dim)
        hidden_states = self.out_proj(out)

        if analysis_cfg is not None and stats is not None and attn_gate_threshold is None:
            select_gate = torch.exp(cache.log_gate)
            K = select_gate.shape[2]
            scale = (select_gate > 0).sum(dim=-1, keepdim=True).float()
            scaled_weight = attn_weights * scale.unsqueeze(2)
            key_valid_mask = (select_gate > 0).unsqueeze(2)
            scaled_weight = scaled_weight * key_valid_mask
            sum_per_key = scaled_weight.sum(dim=2)
            count_per_key = (scaled_weight > 0).sum(dim=2)
            num_bins = analysis_cfg.gate_attn_num_bins
            gate_vals = select_gate

            bin_idx = (gate_vals * num_bins).long().clamp(0, num_bins - 1)
            head_idx = torch.arange(self.num_heads, device=device).view(1, self.num_heads, 1).expand(batch_size, -1, K).reshape(-1)
            flat_index = head_idx * num_bins + bin_idx.reshape(-1)

            step_attn_mass = torch.zeros(self.num_heads * num_bins, dtype=torch.float32, device=device)
            step_attn_count = torch.zeros_like(step_attn_mass)
            step_gate_freq = torch.zeros_like(step_attn_mass)

            step_attn_mass.scatter_add_(0, flat_index, sum_per_key.reshape(-1))
            step_attn_count.scatter_add_(0, flat_index, count_per_key.reshape(-1).float())
            gate_ones = torch.ones_like(flat_index, dtype=torch.float32)
            step_gate_freq.scatter_add_(0, flat_index, gate_ones)

            key = "cross_attn_gate_analysis"
            stats[key]["attn_mass"] += step_attn_mass.view(self.num_heads, num_bins)
            stats[key]["attn_count"] += step_attn_count.view(self.num_heads, num_bins)
            stats[key]["gate_freq"] += step_gate_freq.view(self.num_heads, num_bins)

        return hidden_states