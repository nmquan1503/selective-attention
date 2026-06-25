import torch

class SelectiveAttnCache:
    def __init__(self):
        self.k_rot = None   # (batch_size, num_heads, compressed_len, head_dim)
        self.v = None   # (batch_size, num_heads, compressed_len, head_dim)
        self.gate = None    # (batch_size, num_heads, compressed_len)
        self.valid_mask = None  # (batch_size, num_heads, compressed_len)
        self.write_idx = 0

    def build_kv(
        self,
        k_rot: torch.Tensor,
        v: torch.Tensor,
        gate: torch.Tensor,
        kept_mask: torch.Tensor
    ):
        """
        Args:
            k_rot: (batch_size, num_heads, compressed_len, head_dim)
            v: (batch_size, num_heads, compressed_len, head_dim)
            gate: (batch_size, num_heads, compressed_len)
            kept_mask: (batch_size, num_heads, compressed_len)
        """
        self.k_rot = k_rot.contiguous()
        self.v = v.contiguous()
        self.gate = gate.contiguous()
        self.valid_mask = kept_mask.contiguous()
        self.write_idx = self.k_rot.shape[2]

    def update_kv(
        self,
        k_rot: torch.Tensor,
        v: torch.Tensor,
        gate: torch.Tensor,
        kept_mask: torch.Tensor
    ):
        """
        Args:
            k_rot: (batch_size, self.num_heads, self.head_dim)
            v: (batch_size, self.num_heads, self.head_dim)
            gate: (batch_size, num_heads)
            kept_mask: (batch_size, num_heads)
        """
        self.valid_mask[:, :, self.write_idx] = kept_mask
        self.k_rot[:, :, self.write_idx, :] = k_rot
        self.v[:, :, self.write_idx, :] = v
        self.gate[:, :, self.write_idx] = gate
        self.write_idx += 1