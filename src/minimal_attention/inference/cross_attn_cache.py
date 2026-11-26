import torch

class CrossAttnCache:
    def __init__(self):
        self.k_rot = None   # (batch_size, num_heads, seq_len, head_dim)
        self.v = None   # (batch_size, num_heads, seq_len, head_dim)
        self.mask = None  # (batch_size, seq_len)

    def build(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: torch.Tensor,
    ):
        """
        Args:
            k_rot: (batch_size, num_heads, seq_len, head_dim)
            v: (batch_size, num_heads, seq_len, head_dim)
            mask: (batch_size, seq_len)
        """
        self.k = k
        self.v = v
        self.mask = mask