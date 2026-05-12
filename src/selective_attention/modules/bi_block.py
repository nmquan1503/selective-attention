import torch
import torch.nn as nn

from .ssm import SSM
from .selective_attention import SelectiveMHA
from .feed_forward import SwiGLU
from .rms_norm import RMSNorm

class BiBlock(nn.Module):
    def __init__(
        self,
        model_dim: int = 512,
        head_dim: int = 64,
        ssm_state_dim: int = 128,
        ssm_conv_kernel_size: int = 4,
        ssm_num_groups: int = 1,
        ssm_chunk_size: int = 256,
        mlconv_radius: int = 2,
        dropout_rate: float = 0.15,
        device="cuda"
    ):
        super().__init__()

        self.norm1 = RMSNorm(model_dim)
        self.norm2 = RMSNorm(model_dim)
        self.norm3 = RMSNorm(model_dim)
        self.ssm = SSM(
            model_dim=model_dim,
            state_dim=ssm_state_dim,
            conv_kernel_size=ssm_conv_kernel_size,
            head_dim=head_dim,
            num_groups=ssm_num_groups,
            chunk_size=ssm_chunk_size,
            dropout_rate=dropout_rate,
            device=device
        )
        self.gate_conv = nn.Conv1d(
            in_channels=model_dim,
            out_channels=1,
            kernel_size=2 * mlconv_radius + 1,
            padding=mlconv_radius
        )
        self.mha = SelectiveMHA(model_dim, head_dim)
        self.ffn = SwiGLU(model_dim, model_dim * 4)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(
        self, 
        hidden_states: torch.Tensor, 
        lengths: torch.Tensor | None = None
    ):
        """
        Args:
            hidden_states: (batch_size, seq_len, model_dim)
            lengths: (batch_size,)
        
        Returns: 
            hidden_states: (batch_size, seq_len, model_dim)
            last_ssm_hiddens: (batch_size, inner_dim, state_dim)
        """

        res = hidden_states
        hidden_states = self.norm1(hidden_states)
        hidden_states, last_ssm_hiddens = self.ssm(
            hidden_states=hidden_states, 
            lengths=lengths, 
        )
        hidden_states = res + self.dropout(hidden_states)

        res = hidden_states
        hidden_states = self.norm2(hidden_states)
        gate = torch.sigmoid(self.gate_conv(
            hidden_states.transpose(1, 2)
        ).squeeze(1))
        hidden_states = self.mha(
            hidden_states=hidden_states, 
            gate=gate, 
            lengths=lengths
        )
        hidden_states = res + self.dropout(hidden_states)

        res = hidden_states
        hidden_states = self.norm3(hidden_states)
        hidden_states = self.ffn(hidden_states)
        hidden_states = res + self.dropout(hidden_states)
        
        return hidden_states, last_ssm_hiddens
