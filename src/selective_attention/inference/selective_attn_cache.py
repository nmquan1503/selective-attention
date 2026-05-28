import torch
import math

def _right_align(
    x: torch.Tensor,
    valid_mask: torch.Tensor,
    buffer_size: int
):
    """
    Args:
        x: (batch_size, num_heads, seq_len, head_dim)
        valid_mask: (batch_size, seq_len)
    
    Returns:
        x: (batch_size, num_heads, compressed_len + buffer_size, head_dim)
    """
    batch_size, num_heads, seq_len, head_dim = x.shape
    device = x.device

    num_valid = valid_mask.sum(dim=1)
    max_valid = num_valid.max().item()
    rank = valid_mask.cumsum(dim=1) - 1
    shift = max_valid - num_valid
    out = torch.empty(
        batch_size, num_heads, max_valid + buffer_size, head_dim,
        device=device, dtype=x.dtype
    )
    src_valid = x.permute(0, 2, 1, 3)[valid_mask]
    batch_idx = torch.arange(batch_size, device=device)
    batch_idx = batch_idx.repeat_interleave(num_valid)
    dst_valid = rank[valid_mask] + shift.repeat_interleave(num_valid)
    out[batch_idx, :, dst_valid, :] = src_valid
    valid_mask = (
        torch.arange(max_valid + buffer_size, device=device)
        .unsqueeze(0) 
        >= shift.unsqueeze(1)
    ) & (
        torch.arange(max_valid + buffer_size, device=device)
        .unsqueeze(0)
        < (shift + num_valid).unsqueeze(1)
    )
    return out, valid_mask

class SelectiveAttnCache:
    def __init__(self):
        self.k_rot = None   # (batch_size, num_heads, compressed_len, head_dim)
        self.v = None   # (batch_size, num_heads, compressed_len, head_dim)
        self.write_idx = 0
        self.valid_mask = None  # (batch_size, compressed_len)

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
        hard_gate: torch.Tensor,
        lengths: torch.Tensor
    ):
        """
        Args:
            k_rot: (batch_size, num_heads, seq_len, head_dim)
            v: (batch_size, num_heads, seq_len, head_dim)
            hard_gate: (batch_size, seq_len)
            lengths: (batch_size)
        """
        batch_size, seq_len = hard_gate.shape
        device = k_rot.device

        self.k_rot = k_rot
        self.v = v
        
        pos = torch.arange(seq_len, device=device).unsqueeze(0)
        self.valid_mask = pos < lengths.unsqueeze(1)
        self.valid_mask = self.valid_mask & (hard_gate == 1)

    def reset(self, buffer_size: int):
        self.k_rot, _ = _right_align(
            self.k_rot,
            self.valid_mask,
            buffer_size=buffer_size
        )
        self.v, self.valid_mask = _right_align(
            self.v,
            self.valid_mask,
            buffer_size=buffer_size
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
        hard_gate: torch.Tensor
    ):
        """
        Args:
            k_rot: (batch_size, self.num_heads, self.head_dim)
            v: (batch_size, self.num_heads, self.head_dim)
            hard_gate: (batch_size,)
        """
        self.valid_mask[:, self.write_idx] = (hard_gate == 1)
        self.k_rot[:, :, self.write_idx, :] = k_rot
        self.v[:, :, self.write_idx, :] = v
        self.write_idx += 1