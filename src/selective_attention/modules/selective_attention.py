import torch
import torch.nn as nn
import torch.nn.functional as F

from ..inference import SelectiveAttnCache, InferenceState, GenerationConfig
from .rope import RoPE
from ..utils.validation import check_required

def _right_align(
    x: torch.Tensor,
    valid_mask: torch.Tensor,
    buffer_size: int = 0,
    return_new_mask: bool = False
):
    """
    Args:
        x:          (batch_size, num_heads, seq_len, head_dim)
        valid_mask: (batch_size, num_heads, seq_len)

    Returns:
        x:          (batch_size, num_heads, max_valid + buffer_size, head_dim)
        valid_mask: (batch_size, num_heads, max_valid + buffer_size)
    """
    batch_size, num_heads, seq_len, head_dim = x.shape
    device = x.device

    num_valid = valid_mask.sum(dim=-1)
    max_valid = num_valid.max().item()
    rank = valid_mask.cumsum(dim=-1) - 1
    shift = max_valid - num_valid
    new_len = max_valid + buffer_size

    out = torch.zeros(batch_size, num_heads, new_len, head_dim, device=device, dtype=x.dtype)

    flat_bh = batch_size * num_heads
    out_flat = out.view(flat_bh, new_len, head_dim)
    x_flat = x.view(flat_bh, seq_len, head_dim)
    valid_flat = valid_mask.view(flat_bh, seq_len)
    dst_idx = rank + shift.unsqueeze(-1)
    dst_flat = dst_idx.view(flat_bh, seq_len)

    sel_idx = dst_flat[valid_flat]
    bh_idx = torch.arange(flat_bh, device=device).unsqueeze(-1).expand(-1, seq_len)
    bh_idx_sel = bh_idx[valid_flat]

    out_flat[bh_idx_sel, sel_idx] = x_flat[valid_flat]

    if not return_new_mask:
        return out

    shift_flat = shift.view(flat_bh, 1)
    num_valid_flat = num_valid.view(flat_bh, 1)
    pos_idx = torch.arange(new_len, device=device)

    new_valid_flat = (pos_idx >= shift_flat) & (pos_idx < shift_flat + num_valid_flat)
    new_valid = new_valid_flat.view(batch_size, num_heads, new_len)

    return out, new_valid

def _pad_buffer(x: torch.Tensor, buffer_size: int):
    """
    Args:
        x: (batch_size, num_heads, seq_len, head_dim)
    
    Returns:
        x: (batch_size, num_heads, seq_len + buffer_size, head_dim)
    """
    if buffer_size <= 0:
        return x
    return F.pad(
        x, 
        pad=(
            0, 0,
            0, buffer_size,
            0, 0
        ),
        mode="constant",
        value=0
    )

def _reset_cache(cache: SelectiveAttnCache, buffer_size: int):
    all_kept = cache.valid_mask.all(dim=-1)
    if not all_kept.any():
        cache.gate = _right_align(cache.gate.unsqueeze(-1), cache.valid_mask, buffer_size=buffer_size)
        cache.gate = cache.gate.squeeze(-1)
        cache.k_rot = _right_align(cache.k_rot, cache.valid_mask, buffer_size=buffer_size)
        cache.v, cache.valid_mask = _right_align(cache.v, cache.valid_mask, buffer_size=buffer_size, return_new_mask=True)
    else:
        cache.gate = _pad_buffer(cache.gate.unsqueeze(-1), buffer_size)
        cache.gate = cache.gate.squeeze(-1)
        cache.k_rot = _pad_buffer(cache.k_rot, buffer_size)
        cache.v = _pad_buffer(cache.v, buffer_size)
        cache.valid_mask = _pad_buffer(cache.valid_mask.unsqueeze(-1), buffer_size)
        cache.valid_mask = cache.valid_mask.squeeze(-1)
    if cache.k_rot is not None:
        cache.write_idx = cache.k_rot.shape[2] - buffer_size
    else:
        cache.write_idx = 0

def _build_attn_matrix(
    q: torch.Tensor,
    k: torch.Tensor,
    scale: float,
    kept_mask: torch.Tensor | None = None,
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
        all_kept = kept_mask.all(dim=-1)
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
            k = _right_align(k, kept_mask, buffer_size=1)
            key_pos = _right_align(pos_idx.unsqueeze(-1), kept_mask, buffer_size=1).squeeze(-1).unsqueeze(2)

            attn_matrix = torch.matmul(q, k.transpose(-2, -1))
            k = k[:, :, :-1, :]
            attn_matrix[:, :, :, -1] = self_score
            attn_matrix.mul_(scale)
            
            if is_causal:
                causal_mask = (key_pos >= pos_idx.unsqueeze(-1)) | (key_pos <= 0)
                attn_mask = causal_mask
                attn_mask[:, :, :, -1] = False
            else:
                self_mask = (key_pos == pos_idx.unsqueeze(-1)) & (key_pos > 0)
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
        check_required(cache, "cache", self.is_causal and is_infer, "causal attention inference")
        check_required(lengths, "lengths", not self.is_causal, "non causal attention training and inference")
        
        if is_infer:
            if attn_gate_threshold is None:
                attn_gate_threshold = 0.0

        gate = torch.sigmoid(self.gate_proj(hidden_states))
        select_gate, out_gate = torch.split(gate, [self.num_heads, self.dim], dim=-1)
        select_gate = select_gate.transpose(1, 2).contiguous()
        if is_infer:
            select_gate[select_gate < attn_gate_threshold] = 0.0

        if lengths is not None:
            pad_mask = torch.arange(seq_len, device=device).unsqueeze(0) >= lengths.unsqueeze(1)
            select_gate.masked_fill_(pad_mask.unsqueeze(1), 0.0)

        v = self.v_proj(hidden_states)
        
        kept_mask = None
        if is_infer:
            kept_mask = (select_gate >= attn_gate_threshold) & (select_gate > 0.0)
            if not kept_mask.any():
                hidden_states = self.out_proj(v * out_gate)
                return hidden_states

        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)

        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

        positions = torch.arange(seq_len, device=device)
        q_rot, k_rot = self.rope(q, k, positions, mode="seq")

        attn_matrix, k_rot, is_compressed = _build_attn_matrix(q_rot, k_rot, self.scale, kept_mask, self.is_causal, is_infer)
        if is_compressed:
            select_gate = _right_align(select_gate.unsqueeze(-1), kept_mask).squeeze(-1)
        attn_weight = _gated_softmax(attn_matrix, select_gate, mode="seq", is_infer=is_infer, is_compressed=is_compressed)

        if is_compressed:
            v_aligned, kept_mask = _right_align(v, kept_mask, buffer_size=0, return_new_mask=True)
            out = torch.matmul(attn_weight[...,:-1], v_aligned) + attn_weight[..., -1:] * v
        else:
            v_aligned = v
            out = attn_weight @ v
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.dim)
        hidden_states = self.out_proj(out * out_gate)
        
        if is_infer and self.is_causal:
            cache.build_kv(k_rot, v_aligned, select_gate, kept_mask)
        
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
                cache.valid_mask = torch.zeros((batch_size, self.num_heads, gen_cfg.cache_update_interval), device=device, dtype=torch.bool)
            else:
                _reset_cache(cache, gen_cfg.cache_update_interval)
        
        gate = torch.sigmoid(self.gate_proj(hidden_states))
        select_gate, out_gate = torch.split(gate, [self.num_heads, self.dim], dim=-1)
        kept_mask = (select_gate >= attn_gate_threshold) & (select_gate > 0.0)
        select_gate = select_gate.masked_fill(~kept_mask, 0.0)

        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        q = q.view(batch_size, self.num_heads, self.head_dim)
        k = k.view(batch_size, self.num_heads, self.head_dim)
        v = v.view(batch_size, self.num_heads, self.head_dim)
        
        q_rot, k_rot = self.rope(q, k, state.lengths, mode="pos")
        
        cache.update_kv(k_rot, v, select_gate, kept_mask)

        if cache.write_idx <= 1:
            return self.out_proj(v.view(batch_size, -1) * out_gate)

        cached_k = cache.k_rot[:, :, :cache.write_idx, :] 
        cached_v = cache.v[:, :, :cache.write_idx, :] 
        cached_gate = cache.gate[:, :, :cache.write_idx]
        valid_mask = cache.valid_mask[:, :, :cache.write_idx]

        attn_matrix = (q_rot.unsqueeze(2) @ cached_k.transpose(-2, -1)).squeeze(2)
        attn_matrix.mul_(self.scale)
        attn_matrix[:, :, :-1].masked_fill_(~valid_mask[:, :, :-1], float("-inf"))

        attn_weight = _gated_softmax(attn_matrix, cached_gate, mode="pos", is_infer=True)

        out = attn_weight.unsqueeze(2) @ cached_v
        out = out.squeeze(2).view(batch_size, self.dim)

        if not kept_mask.any():
            cache.write_idx -= 1

        return self.out_proj(out * out_gate)
