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
        device=device, dtype=torch.float32
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
        self.log_gate = None    # (batch_size, compressed_len)
        self.lengths = None # (batch_size,)
        self.write_idx = 0
        self.valid_mask = None  # (batch_size, compressed_len)

    def build(
        self,
        k_rot: torch.Tensor,
        v: torch.Tensor,
        log_gate: torch.Tensor,
        lengths: torch.Tensor
    ):
        """
        Args:
            k_rot: (batch_size, num_heads, seq_len, head_dim)
            v: (batch_size, num_heads, seq_len, head_dim)
            log_gate: (batch_size, mlconv_radius + 1, seq_len)
            lengths: (batch_size)
        """
        batch_size, _, seq_len = log_gate.shape
        mlconv_radius = log_gate.size(1) - 1
        device = k_rot.device

        self.k_rot = k_rot
        self.v = v
        self.lengths = lengths
        
        pos = torch.arange(seq_len, device=device).unsqueeze(0)
        level_idx = torch.clamp(
            pos - (lengths.unsqueeze(1) - 1 - mlconv_radius),
            min=0,
            max=mlconv_radius
        )
        self.log_gate = log_gate[
            torch.arange(batch_size, device=device).unsqueeze(1),
            level_idx,
            pos
        ]
    
    def reset(
        self, 
        mlconv_radius: int,
        gate_threshold: float,
        buffer_size: int,
    ):
        device = self.k_rot.device

        if self.valid_mask is None:
            compressed_len = self.v.shape[2]
            pos = torch.arange(compressed_len, device=device).unsqueeze(0)
            valid_mask = pos < self.lengths.unsqueeze(1)
            gate_mask = self.log_gate >= math.log(gate_threshold)
            tail_start = self.lengths - mlconv_radius
            tail_mask = pos >= tail_start.unsqueeze(1)
            self.valid_mask = valid_mask & (gate_mask | tail_mask)
        
        self.log_gate, _ = _right_align(
            self.log_gate.unsqueeze(1).unsqueeze(-1),
            self.valid_mask,
            buffer_size=buffer_size
        )
        self.log_gate = self.log_gate.squeeze(-1).squeeze(1)

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

    def update(
        self,
        k_rot: torch.Tensor,
        v: torch.Tensor,
        log_gate: torch.Tensor,
        gate_threshold: float
    ):
        """
        Args:
            k_rot: (batch_size, self.num_heads, self.head_dim)
            v: (batch_size, self.num_heads, self.head_dim)
            log_gate: (batch_size, mlconv_radius + 1)
        """
        mlconv_radius = log_gate.size(1) - 1

        remove_mask = log_gate[:, 0] < math.log(gate_threshold)
        self.valid_mask[remove_mask, self.write_idx-mlconv_radius] = False
        self.valid_mask[:, self.write_idx] = True
        
        self.log_gate[:, self.write_idx-mlconv_radius : self.write_idx+1] = log_gate
        self.k_rot[:, :, self.write_idx, :] = k_rot
        self.v[:, :, self.write_idx, :] = v
        self.lengths += 1
        self.write_idx += 1

