import torch
import math
import torch.nn.functional as F

def _right_align(
    x: torch.Tensor,
    valid_mask: torch.Tensor,
    buffer_size: int,
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

class SelectiveAttnCache:
    def __init__(self):
        self.k_rot = None   # (batch_size, num_heads, compressed_len, head_dim)
        self.v = None   # (batch_size, num_heads, compressed_len, head_dim)
        self.write_idx = 0
        self.valid_mask = None  # (batch_size, num_heads, compressed_len)

    def build_conv_ctx(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor,
        cache_size: int,
    ):
        """
        Args:
            x: (batch_size, dim, seq_len)
            lengths: (batch_size,)
        """
        batch_size, dim, seq_len = x.shape
        device = x.device

        cache_idx = torch.arange(cache_size, device=device)[None, None, :]
        src_idx = lengths[:, None, None] - cache_size + cache_idx
        valid = (src_idx >= 0) * (src_idx < lengths[:, None, None])
        src_idx = src_idx.clamp(0, seq_len - 1)
        src_idx = src_idx.expand(batch_size, dim, cache_size)
        self.conv_ctx = torch.gather(x, dim=2, index=src_idx) * valid.expand(batch_size, dim, cache_size)


    def build_kv(
        self,
        k_rot: torch.Tensor,
        v: torch.Tensor,
        gate: torch.Tensor,
        lengths: torch.Tensor,
        gate_threshold: float
    ):
        """
        Args:
            k_rot: (batch_size, num_heads, seq_len, head_dim)
            v: (batch_size, num_heads, seq_len, head_dim)
            gate: (batch_size, num_heads, seq_len)
            lengths: (batch_size)
        """
        batch_size, _, seq_len = gate.shape
        device = k_rot.device

        self.k_rot = k_rot.contiguous()
        self.v = v.contiguous()
        self.gate = gate.contiguous()
        
        hard_gate = gate >= gate_threshold
        pos = torch.arange(seq_len, device=device).unsqueeze(0).unsqueeze(0)
        self.valid_mask = pos < lengths.unsqueeze(1).unsqueeze(1)
        self.valid_mask = self.valid_mask & hard_gate

        self.reset(0)

    def reset(self, buffer_size: int):
        num_valid = self.valid_mask.sum(dim=-1)
        max_valid = num_valid.max().item()
        if max_valid == self.valid_mask.shape[-1]:
            self.gate = _pad_buffer(self.gate.unsqueeze(-1), buffer_size)
            self.gate = self.gate.squeeze(-1)
            self.k_rot = _pad_buffer(self.k_rot, buffer_size)
            self.v = _pad_buffer(self.v, buffer_size)
            self.valid_mask = _pad_buffer(self.valid_mask.unsqueeze(-1), buffer_size)
            self.valid_mask = self.valid_mask.squeeze(-1)   
        else:
            self.gate = _right_align(
                self.gate.unsqueeze(-1),
                self.valid_mask,
                buffer_size=buffer_size
            )
            self.gate = self.gate.squeeze(-1)
            self.k_rot = _right_align(
                self.k_rot,
                self.valid_mask,
                buffer_size=buffer_size
            )
            self.v, self.valid_mask = _right_align(
                self.v,
                self.valid_mask,
                buffer_size=buffer_size,
                return_new_mask=True
            )
            self.write_idx = self.k_rot.shape[2] - buffer_size

    def update_conv_ctx(self, x: torch.Tensor):
        """
        Args: (batch_size, model_dim)
        """
        self.conv_ctx = torch.roll(self.conv_ctx, shifts=-1, dims=-1)
        self.conv_ctx[:, :, -1] = x
    
    def update_kv(
        self,
        k_rot: torch.Tensor,
        v: torch.Tensor,
        gate: torch.Tensor,
        gate_threshold: float
    ):
        """
        Args:
            k_rot: (batch_size, self.num_heads, self.head_dim)
            v: (batch_size, self.num_heads, self.head_dim)
            hard_gate: (batch_size, num_heads)
        """
        hard_gate = gate >= gate_threshold
        self.valid_mask[:, :, self.write_idx] = hard_gate
        self.k_rot[:, :, self.write_idx, :] = k_rot
        self.v[:, :, self.write_idx, :] = v
        self.gate[:, :, self.write_idx] = gate
        self.write_idx += 1