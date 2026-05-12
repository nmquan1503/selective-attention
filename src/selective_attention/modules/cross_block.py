import torch
import torch.nn as nn

from .ssm import SSM
from .multilevel_conv1d import MultiLevelConv1D
from .selective_attention import SelectiveMHA
from .cross_selective_attention import CrossSelectiveMHA
from .feed_forward import SwiGLU
from .rms_norm import RMSNorm
from ..inference import CrossBlockCache, InferenceState, GenerationConfig

class CrossBlock(nn.Module):
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
        self.norm4 = RMSNorm(model_dim)
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
        self.gate_conv = MultiLevelConv1D(model_dim, 1, mlconv_radius)
        self.mha = SelectiveMHA(model_dim, head_dim)
        self.cross_mha = CrossSelectiveMHA(model_dim, head_dim)
        self.ffn = SwiGLU(model_dim, model_dim * 4)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(
        self, 
        hidden_states: torch.Tensor, 
        context: torch.Tensor,
        context_log_gate: torch.Tensor,
        ssm_hiddens: torch.Tensor,
        lengths: torch.Tensor | None = None,
        cache: CrossBlockCache | None = None
    ):
        """
        Args:
            hidden_states: (batch_size, seq_len, model_dim)
            context: (batch_size, context_len, model_dim)
            context_log_gate: (batch_size, context_len)
            ssm_hiddens: (batch_size, inner_dim, state_dim)
            lengths: (batch_size,)
        
        Returns: 
            hidden_states: (batch_size, seq_len, model_dim)
        """

        res = hidden_states
        hidden_states = self.norm1(hidden_states)
        hidden_states, last_ssm_hiddens = self.ssm(
            hidden_states=hidden_states, 
            lengths=lengths, 
            ssm_hiddens=ssm_hiddens, 
            cache=cache.ssm_cache if cache is not None else None
        )
        hidden_states = res + self.dropout(hidden_states)

        res = hidden_states
        hidden_states = self.norm2(hidden_states)
        gate = torch.sigmoid(self.gate_conv(
            x=hidden_states, 
            lengths=lengths, 
            cache=cache.mlconv_cache if cache is not None else None
        ).squeeze(-1))
        hidden_states = self.mha(
            hidden_states=hidden_states, 
            gate=gate, 
            lengths=lengths, 
            cache=cache.attn_cache if cache is not None else None
        )
        hidden_states = res + self.dropout(hidden_states)

        res = hidden_states
        hidden_states = self.norm3(hidden_states)
        hidden_states = self.cross_mha(
            hidden_states=hidden_states,
            context=context,
            gate=context_log_gate,
            cache=cache.cross_attn_cache if cache is not None else None
        )
        hidden_states = res + self.dropout(hidden_states)

        res = hidden_states
        hidden_states = self.norm4(hidden_states)
        hidden_states = self.ffn(hidden_states)
        hidden_states = res + self.dropout(hidden_states)
        
        return hidden_states

    def step(self, hidden_states: torch.Tensor, cache: CrossBlockCache, state: InferenceState, gen_cfg: GenerationConfig):
        """
        Args: (batch_size, model_dim)
        Returns: (batch_size, model_dim)
        """

        res = hidden_states
        hidden_states = self.norm1(hidden_states)
        hidden_states = self.ssm.step(
            hidden_states=hidden_states, 
            cache=cache.ssm_cache
        )
        hidden_states = res + self.dropout(hidden_states)

        res = hidden_states
        hidden_states = self.norm2(hidden_states)
        gate = torch.sigmoid(self.gate_conv.step(
            hidden_states=hidden_states, 
            cache=cache.mlconv_cache
        ).squeeze(-1))
        hidden_states = self.mha.step(
            hidden_states=hidden_states, 
            gate=gate, 
            cache=cache.attn_cache, 
            state=state, 
            gen_cfg=gen_cfg
        )
        hidden_states = res + self.dropout(hidden_states)

        res = hidden_states
        hidden_states = self.norm3(hidden_states)
        hidden_states = self.cross_mha(hidden_states=hidden_states.unsqueeze(1)).squeeze(1)
        hidden_states = res + self.dropout(hidden_states)

        res = hidden_states
        hidden_states = self.norm3(hidden_states)
        hidden_states = self.ffn(hidden_states)
        hidden_states = res + self.dropout(hidden_states)

        return hidden_states