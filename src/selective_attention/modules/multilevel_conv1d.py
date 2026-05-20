import torch
import torch.nn as nn
import torch.nn.functional as F

from ..inference import MultiLevelConv1DCache

class MultiLevelConv1D(nn.Module):
    def __init__(self, in_channels, out_channels, radius):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.radius = radius

        self.causal_conv1d = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=radius + 1,
            padding=radius
        )
        self.right_linears = nn.ModuleList([
            nn.Linear(in_channels, out_channels, bias=False)
            for _ in range(radius)
        ])

    def forward(
        self, 
        x: torch.Tensor,
        lengths: torch.Tensor | None = None,
        cache: MultiLevelConv1DCache | None = None
    ):
        """
        Args:
            x: (batch_size, seq_len, in_channels)
            lengths: (batch_size,)
            
        Returns:
            out: (batch_size, radius + 1, seq_len, out_channels)    # full -> causal
        """
        batch_size, seq_len, _ = x.shape
        device = x.device
        is_infer = cache is not None

        x = x.transpose(1, 2)
        
        if is_infer:
            cache.build_in_ctx(x, lengths, self.radius)

        base = self.causal_conv1d(x).transpose(1, 2)
        base = base[:, :seq_len]
        x = x.transpose(1, 2)

        outs = [base]
        cur = base
        for i in range(self.radius):
            shifted = x[:, i+1:]
            right = self.right_linears[i](shifted)
            right = F.pad(right, (0, 0, 0, i+1))
            cur = cur + right
            outs.append(cur)

        outs = outs[::-1]

        out = torch.stack(outs, dim=1)

        if is_infer:
            cache.build_out_ctx(out, lengths)
        
        return out

    def step(self, x: torch.Tensor, cache: MultiLevelConv1DCache):
        """
        Args:
            x: (batch_size, in_channels)
        
        Returns:
            out: (batch_size, radius + 1, out_channels)
        """

        cache.update_in_ctx(x)

        base = F.conv1d(
            cache.in_ctx,
            self.causal_conv1d.weight,
            self.causal_conv1d.bias
        ).squeeze(-1)

        cache.update_out_ctx(
            [linear(x) for linear in reversed(self.right_linears)] + [base]
        )

        return cache.out_ctx
