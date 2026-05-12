import torch

class SSMCache:
    def __init__(self):
        self.h = None   # (batch_size, seq_len, num_heads, head_dim)
        self.conv_ctx = None    # (batch_size, conv_dim, conv_kernel - 1)

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

        if lengths is None:
            if seq_len >= cache_size:
                self.conv_ctx = x[:, :, -cache_size:]
            else:
                self.conv_ctx = torch.cat(
                    [
                        torch.zeros(batch_size, dim, cache_size - seq_len, device=device, dtype=torch.float32),
                        x
                    ],
                    dim=2
                )
        else:
            cache_idx = torch.arange(cache_size, device=device)[None, None, :]
            src_idx = lengths[:, None, None] - cache_idx + cache_size
            valid = (src_idx >= 0) * (src_idx < lengths[:, None, None])
            src_idx = src_idx.clamp(0, seq_len - 1)
            src_idx = src_idx.expand(batch_size, dim, cache_size)
            self.conv_ctx = torch.gather(x, dim=2, index=src_idx) * valid.expand(batch_size, dim, cache_size)
