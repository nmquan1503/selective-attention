import torch
import torch.nn as nn

def _rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

class RoPE(nn.Module):
    def __init__(self, dim, base=10000):
        super().__init__()
        
        assert dim % 2 == 0
        self.dim = dim
        self.base = base

        half_dim = dim // 2
        inv_freq = 1.0 / (
            base ** (torch.arange(0, half_dim, dtype=torch.float32) / half_dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def _get_cos_sin(self, positions: torch.Tensor):
        """
        positions: (seq_len,) | (batch_size,)
        """
        angles = positions.unsqueeze(-1) * self.inv_freq.unsqueeze(0)
        cos = torch.cos(angles).repeat_interleave(2, dim=-1)
        sin = torch.sin(angles).repeat_interleave(2, dim=-1)
        return cos, sin

    def forward(
        self, 
        q: torch.Tensor, 
        k: torch.Tensor, 
        positions: torch.Tensor, 
        mode="seq"
    ):
        """
        - seq mode:
            Args:
                q, k: (batch_size, num_heads, seq_len, head_dim)
                positions: (seq_len,)
        
        - pos mode:
            Args:
                q, k: (batch_size, num_heads, head_dim)
                positions: (batch_size,)
        """

        cos, sin = self._get_cos_sin(positions)

        if mode == "seq":
            cos = cos[None, None, :, :]
            sin = sin[None, None, :, :]
        else:
            cos = cos[:, None, :]
            sin = sin[:, None, :]

        q_rot = q * cos + _rotate_half(q) * sin
        k_rot = k * cos + _rotate_half(k) * sin

        return q_rot, k_rot
