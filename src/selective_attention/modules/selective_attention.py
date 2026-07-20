import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict

from ..inference import SelectiveAttnCache, InferenceState, GenerationConfig, AnalysisConfig
from .rope import RoPE
from .rms_norm import RMSNorm
from ..utils.tensor_utils import compress, pad_buffer

def _reset_cache(cache: SelectiveAttnCache, buffer_size: int):
    valid_mask = ~torch.isinf(cache.log_gate)
    all_kept = valid_mask.all(dim=-1)
    if not all_kept.any():
        cache.log_gate = compress(cache.log_gate.unsqueeze(-1), valid_mask, buffer_size=buffer_size, pad_value=float("-inf")).squeeze(-1)
        cache.k_rot = compress(cache.k_rot, valid_mask, buffer_size=buffer_size)
        cache.v = compress(cache.v, valid_mask, buffer_size=buffer_size)
    else:
        cache.log_gate = pad_buffer(cache.log_gate.unsqueeze(-1), buffer_size).squeeze(-1)
        cache.k_rot = pad_buffer(cache.k_rot, buffer_size)
        cache.v = pad_buffer(cache.v, buffer_size)
    if cache.k_rot is not None:
        cache.write_idx = cache.k_rot.shape[2] - buffer_size
    else:
        cache.write_idx = 0

def _build_attn_matrix(
    q: torch.Tensor,
    k: torch.Tensor,
    scale: float,
    valid_mask: torch.Tensor | None = None,
    is_causal: bool = False,
    is_infer: bool = False,
):
    """
    Args:
        q, k: (batch_size, num_heads, seq_len, head_dim)
        kept_mask: (batch_size, num_heads, seq_len)
        pad_mask: (batch_size, num_heads, seq_len)
    
    Returns:
        if TRAINING:
            attn_matrix: (batch_size, num_heads, seq_len, seq_len)
        
        if INFER:
            attn_matrix: (batch_size, num_heads, seq_len, num_keys + 1)
    """
    batch_size, num_heads, seq_len, head_dim = q.shape
    device = q.device
    is_compressed = False

    if is_infer:
        all_kept = valid_mask.all(dim=-1)
        is_compressed = not all_kept.any()
        if not is_compressed:
            attn_matrix = torch.matmul(q, k.transpose(-2, -1))
            attn_matrix.mul_(scale)
            if is_causal:
                causal_mask = torch.triu(
                    torch.ones(seq_len, seq_len, dtype=torch.bool, device=device),
                    diagonal=1
                )
                attn_matrix.masked_fill_(causal_mask, float("-inf"))

        else:
            self_score = (q * k).sum(dim=-1)
            pos_idx = torch.arange(seq_len, device=device, dtype=torch.long).view(1, 1, -1).expand(batch_size, num_heads, seq_len) + 1
            k = compress(k, valid_mask, buffer_size=1)
            key_pos = compress(pos_idx.unsqueeze(-1), valid_mask, buffer_size=1).squeeze(-1).unsqueeze(2)

            attn_matrix = torch.matmul(q, k.transpose(-2, -1))
            k = k[:, :, :-1, :]
            attn_matrix[:, :, :, -1] = self_score
            attn_matrix.mul_(scale)
            
            if is_causal:
                causal_mask = key_pos >= pos_idx.unsqueeze(-1)
                attn_mask = causal_mask
                attn_mask[:, :, :, -1] = False
            else:
                self_mask = key_pos == pos_idx.unsqueeze(-1)
                attn_mask = self_mask        

            attn_matrix.masked_fill_(attn_mask, float("-inf"))
    
    else:
        attn_matrix = torch.matmul(q, k.transpose(-2, -1))
        attn_matrix = attn_matrix * scale

        if is_causal:
            causal_mask = torch.triu(
                torch.ones(seq_len, seq_len, dtype=torch.bool, device=device),
                diagonal=1
            )
            attn_matrix = attn_matrix.masked_fill(causal_mask, float("-inf"))
    
    return attn_matrix, k, is_compressed
    
def _gated_softmax(
    attn_matrix: torch.Tensor,
    log_gate: torch.Tensor,
    mode: str = "seq",
    is_infer: bool = False,
    is_compressed: bool = False
):
    """
    Args:
        if TRAINING:
            attn_matrix: (batch_size, num_heads, seq_len, seq_len)
            log_gate: (batch_size, num_heads, seq_len)

        if INFER:
            SEQ MODE:
                attn_matrix: (batch_size, num_heads, seq_len, num_keys + 1)
                log_gate: (batch_size, num_heads, num_keys)
            POS MODE:
                attn_matrix: (batch_size, num_heads, num_keys)
                log_gate: (batch_size, num_heads, num_keys)
    """
    
    if mode == "pos":
        attn_matrix[..., :-1] += log_gate[..., :-1]
        return F.softmax(attn_matrix, dim=-1)
    
    elif mode == "seq":
        if is_compressed:
            attn_matrix[..., :-1] += log_gate.unsqueeze(2)

        else:
            seq_len = attn_matrix.shape[-1]
            device = attn_matrix.device

            diag = torch.arange(seq_len, device=device)
            if is_infer:
                attn_matrix_diag = attn_matrix[:, :, diag, diag].clone()
                attn_matrix += log_gate.unsqueeze(2)
                attn_matrix[:, :, diag, diag] = attn_matrix_diag
            
            else:
                attn_matrix_diag = attn_matrix[:, :, diag, diag]
                attn_matrix = attn_matrix + log_gate.unsqueeze(2)
                attn_matrix[:, :, diag, diag] = attn_matrix_diag

        return F.softmax(attn_matrix, dim=-1)

class SelectiveMHA(nn.Module):
    def __init__(
        self,
        layer_idx: int, 
        dim: int, 
        head_dim: int,
        log_gate_penalty: float,
        is_causal: bool
    ):
        super().__init__()
        self.layer_idx = layer_idx

        assert dim % head_dim == 0
        self.dim = dim
        self.num_heads = dim // head_dim
        self.head_dim = head_dim
        self.is_causal = is_causal
        self.scale = self.head_dim ** -0.5
        self.log_gate_penalty = log_gate_penalty

        self.gate_proj = nn.Linear(dim, self.num_heads + dim)
        self.rope = RoPE(self.head_dim)
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        
        self.register_buffer("score_std_ema", torch.zeros(self.num_heads))

        nn.init.constant_(self.gate_proj.bias, 2.0)

    def forward(
        self, 
        hidden_states: torch.Tensor, 
        lengths: torch.Tensor | None = None,
        attn_gate_threshold: torch.Tensor | None = None,
        cache: SelectiveAttnCache | None = None,
        analysis_cfg: AnalysisConfig | None = None,
        stats: Dict | None = None
    ):
        """
        Args:
            hidden_states: (batch_size, seq_len, dim)
            lengths: (batch_size,)
            attn_gate_threshold: (num_heads,)
        
        Returns:
            hidden_states: (batch_size, seq_len, dim)
        """
        batch_size, seq_len, _ = hidden_states.shape
        device = hidden_states.device
        is_infer = not self.training
        is_prefill = is_infer and self.is_causal and cache is not None

        gate = torch.sigmoid(self.gate_proj(hidden_states))
        select_gate, out_gate = torch.split(gate, [self.num_heads, self.dim], dim=-1)
        select_gate = select_gate.transpose(1, 2).contiguous()
        if is_infer and attn_gate_threshold is not None:
            select_gate[select_gate < attn_gate_threshold[None, :, None]] = 0.0

        if lengths is not None:
            pad_mask = torch.arange(seq_len, device=device).unsqueeze(0) >= lengths.unsqueeze(1)

        v = self.v_proj(hidden_states)
        
        valid_mask = None
        if is_infer:
            valid_mask = select_gate > 0.0
            if attn_gate_threshold is not None:
                valid_mask &= (select_gate >= attn_gate_threshold[None, :, None])
            if lengths is not None:
                valid_mask &= ~pad_mask[:, None, :]
            if not valid_mask.any():
                hidden_states = self.out_proj(v * out_gate)
                return hidden_states

        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)

        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

        positions = torch.arange(seq_len, device=device)
        q_rot, k_rot = self.rope(q, k, positions, mode="seq")

        attn_matrix, k_rot, is_compressed = _build_attn_matrix(q_rot, k_rot, self.scale, valid_mask, self.is_causal, is_infer)
        
        if self.training:
            valid_attention_score_mask = attn_matrix != float("-inf")
            if lengths is not None:
                valid_attention_score_mask = valid_attention_score_mask & ~pad_mask[:, None, :, None] & ~pad_mask[:, None, None, :]
            valid_attention_score_count = valid_attention_score_mask.sum(dim=(0, 2, 3)).float()
            valid_attention_scores = attn_matrix.masked_fill(
                ~valid_attention_score_mask, 0.0
            )
            attention_score_sum = valid_attention_scores.sum(dim=(0, 2, 3))
            attention_score_squared_sum = valid_attention_scores.square().sum(dim=(0, 2, 3))
            head_attention_score_mean = (
                attention_score_sum / valid_attention_score_count.clamp(min=1)
            )
            head_attention_score_variance = (
                attention_score_squared_sum / valid_attention_score_count.clamp(min=1)
                - head_attention_score_mean.square()
            )
            head_attention_score_std = head_attention_score_variance.clamp(min=0).sqrt()
            with torch.no_grad():
                self.score_std_ema.mul_(0.9).add_(
                    head_attention_score_std,
                    alpha=0.1,
                )
        
        log_select_gate = torch.log(select_gate)
        if is_infer:
            log_select_gate *= self.score_std_ema[None, :, None] * self.log_gate_penalty
        else:
            log_select_gate *= head_attention_score_std[None, :, None] * self.log_gate_penalty
        
        if lengths is not None:
            if is_infer:
                log_select_gate.masked_fill_(pad_mask.unsqueeze(1), float("-inf"))
            else:
                log_select_gate = log_select_gate.masked_fill(pad_mask.unsqueeze(1), float("-inf"))

        if is_compressed:
            log_select_gate = compress(log_select_gate.unsqueeze(-1), valid_mask, pad_value=float("-inf")).squeeze(-1)
        
        attn_weight = _gated_softmax(attn_matrix, log_select_gate, mode="seq", is_infer=is_infer, is_compressed=is_compressed)

        if is_compressed:
            v_aligned = compress(v, valid_mask)
            out = torch.matmul(attn_weight[...,:-1], v_aligned) + attn_weight[..., -1:] * v
        else:
            v_aligned = v
            out = attn_weight @ v
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.dim)
        hidden_states = self.out_proj(out * out_gate)
        
        if is_prefill:
            cache.build_kv(k_rot, v_aligned, log_select_gate)
        
        if analysis_cfg is not None and stats is not None  and attn_gate_threshold is None:
            mean_out_gate = out_gate.view(batch_size, seq_len, self.num_heads, self.head_dim).mean(dim=-1).transpose(1, 2)
            query_valid = (mean_out_gate >= 0.1).float()
            if self.is_causal:
                scale = torch.arange(1, seq_len + 1, device=device).view(1, 1, seq_len, 1)
            elif lengths is not None:
                scale = lengths.view(-1, 1, 1, 1)
            else:
                scale = torch.tensor(seq_len, device=device, dtype=attn_weight.dtype)
            scaled_weight = attn_weight * scale * query_valid.unsqueeze(-1)
            pos = torch.arange(seq_len, device=device)
            if lengths is not None:
                query_valid = (pos.unsqueeze(0) < lengths.unsqueeze(1)).unsqueeze(-1) 
                key_valid   = (pos.unsqueeze(0) < lengths.unsqueeze(1)).unsqueeze(1)
                padding_mask = query_valid & key_valid 
                scaled_weight = scaled_weight * padding_mask.unsqueeze(1)

            if self.is_causal:
                causal_valid = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device))
                scaled_weight = scaled_weight * causal_valid.unsqueeze(0).unsqueeze(1)

            self_mask = ~torch.eye(seq_len, dtype=torch.bool, device=device)
            scaled_weight = scaled_weight * self_mask.unsqueeze(0).unsqueeze(0)

            sum_per_key = scaled_weight.sum(dim=2)
            count_per_key = (scaled_weight > 0).sum(dim=2)

            num_bins = analysis_cfg.gate_attn_num_bins
            gate_vals = select_gate
            bin_idx = (gate_vals * num_bins).long().clamp(0, num_bins - 1)
            head_idx = torch.arange(self.num_heads, device=device).view(1, self.num_heads, 1).expand(batch_size, -1, seq_len).reshape(-1)
            flat_index = head_idx * num_bins + bin_idx.reshape(-1)
            
            bin_attn_mass_flat = torch.zeros(self.num_heads * num_bins, dtype=torch.float32, device=device)
            bin_attn_count_flat = torch.zeros(self.num_heads * num_bins, dtype=torch.float32, device=device)
            bin_gate_freq_flat = torch.zeros(self.num_heads * num_bins, dtype=torch.float32, device=device)

            bin_attn_mass_flat.scatter_add_(0, flat_index, sum_per_key.reshape(-1))
            bin_attn_count_flat.scatter_add_(0, flat_index, count_per_key.reshape(-1).float())
            gate_one = torch.ones_like(flat_index, dtype=torch.float32)
            bin_gate_freq_flat.scatter_add_(0, flat_index, gate_one)
            
            bin_attn_mass = bin_attn_mass_flat.view(self.num_heads, num_bins)
            bin_attn_count = bin_attn_count_flat.view(self.num_heads, num_bins)
            bin_gate_freq = bin_gate_freq_flat.view(self.num_heads, num_bins)
                
            stats[f"{'causal' if self.is_causal else 'non_causal'}_attn_gate_analysis"] = {
                "attn_mass": bin_attn_mass,
                "attn_count": bin_attn_count,
                "gate_freq": bin_gate_freq
            }

        return hidden_states

    def step(
        self, 
        hidden_states: torch.Tensor, 
        cache: SelectiveAttnCache, 
        state: InferenceState,
        gen_cfg: GenerationConfig,
        analysis_cfg: AnalysisConfig | None = None,
        stats: Dict | None = None
    ):
        """
        Args:
            hidden_states: (batch_size, model_dim)
        
        Returns:
            hidden_states: (batch_size, model_dim)
        """

        batch_size, _ = hidden_states.shape
        device = hidden_states.device
        attn_gate_threshold = gen_cfg.attn_gate_thresholds[self.layer_idx] if gen_cfg.attn_gate_thresholds is not None else None

        if state.step % gen_cfg.cache_update_interval == 0:
            if cache.k_rot is None:
                cache.k_rot = torch.empty((batch_size, self.num_heads, gen_cfg.cache_update_interval, self.head_dim), device=device, dtype=torch.float32)
                cache.v = torch.empty((batch_size, self.num_heads, gen_cfg.cache_update_interval, self.head_dim), device=device, dtype=torch.float32)
                cache.log_gate = torch.zeros((batch_size, self.num_heads, gen_cfg.cache_update_interval), device=device, dtype=torch.float32)
            else:
                _reset_cache(cache, gen_cfg.cache_update_interval)
        
        gate = torch.sigmoid(self.gate_proj(hidden_states))
        select_gate, out_gate = torch.split(gate, [self.num_heads, self.dim], dim=-1)

        if attn_gate_threshold is not None:
            valid_mask = (select_gate >= attn_gate_threshold[None, :]) & (select_gate > 0.0)
        else:
            valid_mask = select_gate > 0.0
        select_gate = select_gate * valid_mask
        log_select_gate = torch.log(select_gate) * self.score_std_ema[None, :] * self.log_gate_penalty

        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        q = q.view(batch_size, self.num_heads, self.head_dim)
        k = k.view(batch_size, self.num_heads, self.head_dim)
        v = v.view(batch_size, self.num_heads, self.head_dim)
        
        q_rot, k_rot = self.rope(q, k, state.lengths, mode="pos")
        
        cache.update_kv(k_rot, v, log_select_gate)

        if cache.write_idx <= 1:
            return self.out_proj(v.view(batch_size, -1) * out_gate)

        cached_k = cache.k_rot[:, :, :cache.write_idx, :] 
        cached_v = cache.v[:, :, :cache.write_idx, :] 
        cached_log_gate = cache.log_gate[:, :, :cache.write_idx]

        attn_matrix = (q_rot.unsqueeze(2) @ cached_k.transpose(-2, -1)).squeeze(2)
        attn_matrix.mul_(self.scale)

        attn_weight = _gated_softmax(attn_matrix, cached_log_gate, mode="pos", is_infer=True)

        out = attn_weight.unsqueeze(2) @ cached_v
        out = out.squeeze(2).view(batch_size, self.dim)

        if not valid_mask.any():
            cache.write_idx -= 1

        if analysis_cfg is not None and stats is not None and attn_gate_threshold is None:
            mean_out_gate = out_gate.view(batch_size, self.num_heads, self.head_dim).mean(dim=-1)
            query_valid = (mean_out_gate >= 0.1).float()
            seq_len = cache.write_idx
            scaled_weight = attn_weight * seq_len * query_valid.unsqueeze(-1)

            num_bins = analysis_cfg.gate_attn_num_bins
            attn_mass = stats["causal_attn_gate_analysis"]["attn_mass"]
            attn_count = stats["causal_attn_gate_analysis"]["attn_count"]
            gate_freq = stats["causal_attn_gate_analysis"]["gate_freq"]

            gate_sub = torch.exp(cached_log_gate[:, :, :seq_len-1])
            weight_sub = scaled_weight[:, :, :seq_len-1]

            B, H, S_minus_1 = gate_sub.shape
            bin_idx = (gate_sub * num_bins).long().clamp(0, num_bins - 1)
            head_idx = torch.arange(H, device=device).view(1, H, 1).expand(B, -1, S_minus_1).reshape(-1)
            flat_index = head_idx * num_bins + bin_idx.reshape(-1)

            mask_valid = (gate_sub > 0).float()
            sum_flat = (weight_sub * mask_valid).reshape(-1)
            count_flat = mask_valid.reshape(-1)
            gate_ones = mask_valid.reshape(-1)

            step_attn_mass = torch.zeros(H * num_bins, dtype=torch.float32, device=device)
            step_attn_count = torch.zeros(H * num_bins, dtype=torch.float32, device=device)
            step_gate_freq = torch.zeros(H * num_bins, dtype=torch.float32, device=device)

            step_attn_mass.scatter_add_(0, flat_index, sum_flat)
            step_attn_count.scatter_add_(0, flat_index, count_flat)
            step_gate_freq.scatter_add_(0, flat_index, gate_ones)

            attn_mass += step_attn_mass.view(H, num_bins)
            attn_count += step_attn_count.view(H, num_bins)
            gate_freq += step_gate_freq.view(H, num_bins)

        return self.out_proj(out * out_gate)
