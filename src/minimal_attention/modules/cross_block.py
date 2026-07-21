import torch
import torch.nn as nn

from .ssm import SSM
from .minimal_attention import MinMHA
from .cross_minimal_attention import CrossMinMHA
from .feed_forward import SwiGLU
from .rms_norm import RMSNorm
from ..inference import CrossBlockCache, InferenceState, GenerationConfig, AnalysisConfig

class CrossBlock(nn.Module):
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
        self.norm4 = RMSNorm(model_dim)
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
            is_causal=True
        )
        self.cross_mha = CrossMinMHA(
            layer_idx=layer_idx,
            dim=model_dim, 
            head_dim=head_dim,
            log_gate_penalty=attn_log_gate_penalty
        )
        self.ffn = SwiGLU(model_dim, model_dim * 4)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(
        self, 
        hidden_states: torch.Tensor, 
        context: torch.Tensor,
        context_lengths: torch.Tensor,
        ssm_hiddens: torch.Tensor,
        self_attn_gate_threshold: torch.Tensor | None = None,
        cross_attn_gate_threshold: torch.Tensor | None = None,
        cache: CrossBlockCache | None = None,
        analysis_cfg: AnalysisConfig | None = None,
        stats: dict | None = None
    ):
        """
        Args:
            hidden_states: (batch_size, seq_len, model_dim)
            context: (batch_size, context_len, model_dim)
            context_lengths: (batch_size,)
            ssm_hiddens: (batch_size, inner_dim, state_dim)
            self_attn_gate_threshold: (num_heads,)
            cross_attn_gate_threshold: (num_heads,)
        
        Returns: 
            hidden_states: (batch_size, seq_len, model_dim)
        """

        res = hidden_states
        hidden_states = self.norm1(hidden_states)
        hidden_states, last_ssm_hiddens = self.ssm(
            hidden_states=hidden_states, 
            ssm_hiddens=ssm_hiddens, 
            cache=cache.ssm_cache if cache is not None else None
        )
        hidden_states = res + self.dropout(hidden_states)

        res = hidden_states
        hidden_states = self.norm2(hidden_states)
        hidden_states = self.mha(
            hidden_states=hidden_states, 
            attn_gate_threshold=self_attn_gate_threshold,
            cache=cache.attn_cache if cache is not None else None,
            analysis_cfg=analysis_cfg,
            stats=stats
        )
        hidden_states = res + self.dropout(hidden_states)

        res = hidden_states
        hidden_states = self.norm3(hidden_states)
        hidden_states = self.cross_mha(
            hidden_states=hidden_states,
            context=context,
            context_lengths=context_lengths,
            attn_gate_threshold=cross_attn_gate_threshold,
            cache=cache.cross_attn_cache if cache is not None else None,
            analysis_cfg=analysis_cfg,
            stats=stats
        )
        hidden_states = res + self.dropout(hidden_states)

        res = hidden_states
        hidden_states = self.norm4(hidden_states)
        hidden_states = self.ffn(hidden_states)
        hidden_states = res + self.dropout(hidden_states)
        
        return hidden_states

    def step(
        self,
        hidden_states: torch.Tensor, 
        cache: CrossBlockCache, 
        state: InferenceState, 
        gen_cfg: GenerationConfig,
        analysis_cfg: AnalysisConfig,
        stats: dict | None = None
    ):
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
        hidden_states = self.mha.step(
            hidden_states=hidden_states, 
            cache=cache.attn_cache, 
            state=state, 
            gen_cfg=gen_cfg,
            analysis_cfg=analysis_cfg,
            stats=stats
        )
        hidden_states = res + self.dropout(hidden_states)

        res = hidden_states
        hidden_states = self.norm3(hidden_states)
        hidden_states = self.cross_mha.step(
            hidden_states=hidden_states,
            cache=cache.cross_attn_cache,
            gen_cfg=gen_cfg,
            analysis_cfg=analysis_cfg,
            stats=stats
        )
        hidden_states = res + self.dropout(hidden_states)

        res = hidden_states
        hidden_states = self.norm4(hidden_states)
        hidden_states = self.ffn(hidden_states)
        hidden_states = res + self.dropout(hidden_states)

        return hidden_states