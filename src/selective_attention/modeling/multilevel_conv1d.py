import torch
import torch.nn as nn
import torch.nn.functional as F

from selective_attention.modeling.inference_state import InferenceState

class MultiLevelConv1D(nn.Module):
    def __init__(self, layer_idx, in_channels, out_channels, radius):
        super().__init__()
        self.layer_idx = layer_idx
        
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
            nn.Linear(in_channels, out_channels)
            for _ in range(radius)
        ])

    def forward(
        self, 
        x: torch.Tensor,
        lengths: torch.Tensor | None = None,
        state: InferenceState | None = None
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
        is_infer = state is not None
        if is_infer:
            layer_state = state.layers[self.layer_idx]

        x = x.transpose(1, 2)
        if is_infer:
            idx = lengths[:, None] - (self.radius + 1) + torch.arange(self.radius + 1, device=device)
            layer_state.mlconv_in_ctx = torch.gather(
                x,
                2,
                idx.clamp(min=0)[:, None].expand(-1, self.in_channels, -1)
            ) * (idx >= 0)[:, None]

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
            level_idx = torch.arange(self.radius + 1, device=device)
            batch_idx = torch.arange(batch_size, device=device)[:, None]
            pos = lengths[:, None] - (self.radius + 1) + level_idx
            valid = pos >= 0
            pos = pos.clamp(min=0)
            layer_state.mlconv_out_ctx = (
                out[batch_idx, level_idx[None, :], pos] * valid.unsqueeze(-1)
            )
        
        return out

    def step(self, x: torch.Tensor, state: InferenceState):
        """
        Args:
            x: (batch_size, in_channels)
        
        Returns:
            out: (batch_size, radius + 1, out_channels)
        """
        layer_state = state.layers[self.layer_idx]

        ctx_x = layer_state.mlconv_in_ctx
        ctx_x = torch.roll(ctx_x, shifts=-1, dims=2)
        ctx_x[:, :, -1] = x
        layer_state.mlconv_in_ctx = ctx_x

        base = F.conv1d(
            ctx_x,
            self.causal_conv1d.weight,
            self.causal_conv1d.bias
        ).squeeze(-1)

        out_ctx = layer_state.mlconv_out_ctx

        for i, linear in enumerate(self.right_linears):
            out_ctx[:, -1-i, :] = out_ctx[:, -1-i, :] + linear(x)

        out_ctx = torch.roll(out_ctx, shifts=-1, dims=1)
        out_ctx[:, -1, :] = base

        layer_state.mlconv_out_ctx = out_ctx

        return out_ctx
