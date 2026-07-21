import torch
import torch.nn as nn

from .ssm import SSM
from .minimal_attention import MinMHA
from .feed_forward import SwiGLU
from .rms_norm import RMSNorm
from ..inference import AnalysisConfig

class BiBlock(nn.Module):
    def __init__(
        self,
        layer_idx: int,
        model_dim: int = 512,
        head_dim: int = 64,
        attn_log_gate_penalty: float = 2,
        ssm_state_dim: int = 128,
        ssm_conv_kernel_size: int = 4,
        ssm_num_groups: int = 1,
        ssm_chunk_size: int = 256,
        dropout_rate: float = 0.15,
        device="cuda"
    ):
        super().__init__()
        self.layer_idx = layer_idx

        self.norm1 = RMSNorm(model_dim)
        self.norm2 = RMSNorm(model_dim)
        self.norm3 = RMSNorm(model_dim)
        self.ssm = SSM(
            layer_idx=layer_idx,
            model_dim=model_dim,
            state_dim=ssm_state_dim,
            conv_kernel_size=ssm_conv_kernel_size,
            head_dim=head_dim,
            num_groups=ssm_num_groups,
            chunk_size=ssm_chunk_size,
            dropout_rate=dropout_rate,
            device=device
        )
        self.mha = MinMHA(
            layer_idx=layer_idx, 
            dim=model_dim, 
            head_dim=head_dim,
            log_gate_penalty=attn_log_gate_penalty,
            is_causal=False
        )
        self.ffn = SwiGLU(model_dim, model_dim * 4)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(
        self, 
        hidden_states: torch.Tensor, 
        lengths: torch.Tensor | None = None,
        attn_gate_threshold: torch.Tensor | None = None,
        analysis_cfg: AnalysisConfig | None = None,
        stats: dict | None = None
    ):
        """
        Args:
            hidden_states: (batch_size, seq_len, model_dim)
            lengths: (batch_size,)
            attn_gate_threshold: (num_heads,)
        
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
        hidden_states = self.mha(
            hidden_states=hidden_states, 
            lengths=lengths,
            attn_gate_threshold=attn_gate_threshold,
            analysis_cfg=analysis_cfg,
            stats=stats
        )
        hidden_states = res + self.dropout(hidden_states)

        res = hidden_states
        hidden_states = self.norm3(hidden_states)
        hidden_states = self.ffn(hidden_states)
        hidden_states = res + self.dropout(hidden_states)
        
        return hidden_states, last_ssm_hiddens
