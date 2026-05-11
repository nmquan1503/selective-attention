import torch

class MultiLevelConv1DCache:
    def __init__(self,):
        self.in_ctx = None   # (batch_size, in_channels, radius + 1)
        self.out_ctx = None  # (batch_size, radius + 1, out_channels)

    def build_in_ctx(
        self, 
        x: torch.Tensor, 
        lengths: torch.Tensor,
        radius: int,
    ):
        """
        Args:
            x: (batch_size, in_channels, seq_len)
            lengths: (batch_size,)
        """
        in_channels = x.size(1)
        device = x.device

        idx = lengths[:, None] - (radius + 1) + torch.arange(radius + 1, device=device)
        self.in_ctx = torch.gather(
            x,
            2,
            idx.clamp(min=0)[:, None].expand(-1, in_channels, -1)
        ) * (idx >= 0)[:, None]
    
    def build_out_ctx(
        self,
        out: torch.Tensor,
        lengths: torch.Tensor,
    ):
        """
        Args:
            out: (batch_size, radius + 1, seq_len, out_channels)
            lengths: (batch_size,)
        """

        batch_size = out.size(0)
        radius = out.size(1) - 1
        device = out.device

        level_idx = torch.arange(radius + 1, device=device)
        batch_idx = torch.arange(batch_size, device=device)[:, None]
        pos = lengths[:, None] - (radius + 1) + level_idx
        valid = pos >= 0
        pos = pos.clamp(min=0)
        self.out_ctx = (
            out[batch_idx, level_idx[None, :], pos] * valid.unsqueeze(-1)
        )
    
    def update_in_ctx(
        self,
        x: torch.Tensor,
    ):
        """
        Args: (batch_size, in_channels)
        """
        self.in_ctx = torch.roll(self.in_ctx, shifts=-1, dims=2)
        self.in_ctx[:, :, -1] = x
    
    def update_out_ctx(
        self,
        contribs: list[torch.Tensor],
    ):
        """
        Args:
            contribs: List of (batch_size, out_channels)
        """
        radius = len(contribs) - 1
        self.out_ctx = torch.roll(self.out_ctx, shifts=-1, dims=1)
        for i in range(radius):
            self.out_ctx[:, i, :] += contribs[i]
        self.out_ctx[:, -1, :] = contribs[-1]
