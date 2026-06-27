import torch
import torch.nn as nn
import torch.nn.functional as F

from ..inference import SelectiveAttnCache, InferenceState, GenerationConfig
from .rope import RoPE
from ..utils.tensor_utils import compress, pad_buffer

def _reset_cache(cache: SelectiveAttnCache, buffer_size: int):
    valid_mask = cache.gate > 0.0
    all_kept = valid_mask.all(dim=-1)
    if not all_kept.any():
        cache.gate = compress(cache.gate.unsqueeze(-1), valid_mask, buffer_size=buffer_size).squeeze(-1)
        cache.k_rot = compress(cache.k_rot, valid_mask, buffer_size=buffer_size)
        cache.v = compress(cache.v, valid_mask, buffer_size=buffer_size)
    else:
        cache.gate = pad_buffer(cache.gate.unsqueeze(-1), buffer_size).squeeze(-1)
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
    is_infer: bool = False
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
    gate: torch.Tensor,
    eps: float = 1e-12,
    mode: str = "seq",
    is_infer: bool = False,
    is_compressed: bool = False
):
    """
    Args:
        if TRAINING:
            attn_matrix: (batch_size, num_heads, seq_len, seq_len)
            gate: (batch_size, num_heads, seq_len)

        if INFER:
            SEQ MODE:
                attn_matrix: (batch_size, num_heads, seq_len, num_keys + 1)
                gate: (batch_size, num_heads, num_keys)
            POS MODE:
                attn_matrix: (batch_size, num_heads, num_keys)
                gate: (batch_size, num_heads, num_keys)
    """
    seq_len = attn_matrix.shape[-2]

    max_val = attn_matrix.max(dim=-1, keepdim=True).values
    if is_infer:
        attn_matrix.sub_(max_val)
        attn_matrix.exp_()
        exp_attn = attn_matrix
        if mode == "pos":
            exp_attn[..., :-1].mul_(gate[..., :-1])
            numerator = exp_attn
        else:
            if is_compressed:
                exp_attn[..., :-1].mul_(gate.unsqueeze(2))
                numerator = exp_attn
            else:
                numerator = exp_attn * gate.unsqueeze(2)
                diag = torch.arange(seq_len, device=attn_matrix.device)
                numerator[..., diag, diag] = exp_attn[..., diag, diag]

    else:
        exp_attn = torch.exp(attn_matrix - max_val)
        numerator = exp_attn * gate.unsqueeze(2)
        diag = torch.arange(seq_len, device=attn_matrix.device)
        numerator[..., diag, diag] = exp_attn[..., diag, diag]

    denom = numerator.sum(dim=-1, keepdim=True)
    
    if is_infer:
        numerator.div_(denom + eps)
        attn_weight = numerator
    else:
        attn_weight = numerator / (denom + eps)
    
    return attn_weight

class SelectiveMHA(nn.Module):
    def __init__(
        self,
        layer_idx: int, 
        dim: int, 
        head_dim: int,
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

        self.gate_proj = nn.Linear(dim, self.num_heads + dim)
        self.rope = RoPE(self.head_dim)
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

    def forward(
        self, 
        hidden_states: torch.Tensor, 
        lengths: torch.Tensor | None = None,
        attn_gate_threshold: float | None = None,
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
        is_infer = not self.training
        is_prefill = is_infer and self.is_causal and cache is not None

        gate = torch.sigmoid(self.gate_proj(hidden_states))
        select_gate, out_gate = torch.split(gate, [self.num_heads, self.dim], dim=-1)
        select_gate = select_gate.transpose(1, 2).contiguous()
        if is_infer and attn_gate_threshold is not None:
            select_gate[select_gate < attn_gate_threshold] = 0.0

        if lengths is not None:
            pad_mask = torch.arange(seq_len, device=device).unsqueeze(0) >= lengths.unsqueeze(1)
            if is_infer:
                select_gate.masked_fill_(pad_mask.unsqueeze(1), 0.0)
            else:
                select_gate = select_gate.masked_fill(pad_mask.unsqueeze(1), 0.0)

        v = self.v_proj(hidden_states)
        
        valid_mask = None
        if is_infer:
            valid_mask = select_gate > 0.0
            if attn_gate_threshold is not None:
                valid_mask &= (select_gate >= attn_gate_threshold)
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
        if is_compressed:
            select_gate = compress(select_gate.unsqueeze(-1), valid_mask).squeeze(-1)
        attn_weight = _gated_softmax(attn_matrix, select_gate, mode="seq", is_infer=is_infer, is_compressed=is_compressed)

        if is_compressed:
            v_aligned = compress(v, valid_mask)
            out = torch.matmul(attn_weight[...,:-1], v_aligned) + attn_weight[..., -1:] * v
        else:
            v_aligned = v
            out = attn_weight @ v
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.dim)
        hidden_states = self.out_proj(out * out_gate)
        
        if is_prefill:
            cache.build_kv(k_rot, v_aligned, select_gate)
        
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
        attn_gate_threshold = gen_cfg.attn_gate_thresholds[self.layer_idx]

        if state.step % gen_cfg.cache_update_interval == 0:
            if cache.k_rot is None:
                cache.k_rot = torch.empty((batch_size, self.num_heads, gen_cfg.cache_update_interval, self.head_dim), device=device, dtype=torch.float32)
                cache.v = torch.empty((batch_size, self.num_heads, gen_cfg.cache_update_interval, self.head_dim), device=device, dtype=torch.float32)
                cache.gate = torch.zeros((batch_size, self.num_heads, gen_cfg.cache_update_interval), device=device, dtype=torch.float32)
            else:
                _reset_cache(cache, gen_cfg.cache_update_interval)
        
        gate = torch.sigmoid(self.gate_proj(hidden_states))
        select_gate, out_gate = torch.split(gate, [self.num_heads, self.dim], dim=-1)
        valid_mask = (select_gate >= attn_gate_threshold) & (select_gate > 0.0)
        select_gate *= valid_mask

        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        q = q.view(batch_size, self.num_heads, self.head_dim)
        k = k.view(batch_size, self.num_heads, self.head_dim)
        v = v.view(batch_size, self.num_heads, self.head_dim)
        
        q_rot, k_rot = self.rope(q, k, state.lengths, mode="pos")
        
        cache.update_kv(k_rot, v, select_gate)

        if cache.write_idx <= 1:
            return self.out_proj(v.view(batch_size, -1) * out_gate)

        cached_k = cache.k_rot[:, :, :cache.write_idx, :] 
        cached_v = cache.v[:, :, :cache.write_idx, :] 
        cached_gate = cache.gate[:, :, :cache.write_idx]

        attn_matrix = (q_rot.unsqueeze(2) @ cached_k.transpose(-2, -1)).squeeze(2)
        attn_matrix.mul_(self.scale)

        attn_weight = _gated_softmax(attn_matrix, cached_gate, mode="pos", is_infer=True)

        out = attn_weight.unsqueeze(2) @ cached_v
        out = out.squeeze(2).view(batch_size, self.dim)

        if not valid_mask.any():
            cache.write_idx -= 1

        return self.out_proj(out * out_gate)
