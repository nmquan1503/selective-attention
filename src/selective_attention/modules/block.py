import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple

from .ssm import SSM
from .multilevel_conv1d import MultiLevelConv1D
from .selective_attention import SelectiveMHA
from .feed_forward import SwiGLU
from .rms_norm import RMSNorm
from ..inference import BlockCache, InferenceState, GenerationConfig

class Block(nn.Module):
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
        self.gate_conv = MultiLevelConv1D(model_dim, 1, mlconv_radius)
        self.mha = SelectiveMHA(model_dim, head_dim)
        self.ffn = SwiGLU(model_dim, model_dim * 4)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(
        self, 
        hidden_states: torch.Tensor, 
        lengths: torch.Tensor | None = None,
        ssm_hiddens: torch.Tensor | None = None, 
        conv_context: torch.Tensor | None = None,
        cache: BlockCache | None = None
    ):
        """
        Args:
            hidden_states: (batch_size, seq_len, model_dim)
            lengths: (batch_size,)
            ssm_hiddens: (batch_size, inner_dim, state_dim)
            conv_context: (batch_size, hBC_dim, conv_kernel - 1)
        
        Returns: 
            hidden_states: (batch_size, seq_len, model_dim)
            last_ssm_hiddens: (batch_size, inner_dim, state_dim)
        """

        res = hidden_states
        hidden_states = self.norm1(hidden_states)
        hidden_states, last_ssm_hiddens = self.ssm(hidden_states, lengths, ssm_hiddens, conv_context, cache.ssm_cache)
        hidden_states = res + self.dropout(hidden_states)

        res = hidden_states
        hidden_states = self.norm2(hidden_states)
        gate = torch.sigmoid(self.gate_conv(hidden_states, lengths, cache.mlconv_cache).squeeze(-1))
        hidden_states = self.mha(hidden_states, gate, lengths, cache.attn_cache)
        hidden_states = res + self.dropout(hidden_states)

        res = hidden_states
        hidden_states = self.norm3(hidden_states)
        hidden_states = self.ffn(hidden_states)
        hidden_states = res + self.dropout(hidden_states)
        
        return hidden_states, last_ssm_hiddens

    def step(self, hidden_states: torch.Tensor, cache: BlockCache, state: InferenceState, gen_cfg: GenerationConfig):
        """
        Args: (batch_size, model_dim)
        Returns: (batch_size, model_dim)
        """

        res = hidden_states
        hidden_states = self.norm1(hidden_states)
        hidden_states = self.ssm.step(hidden_states, cache.ssm_cache)
        hidden_states = res + self.dropout(hidden_states)

        res = hidden_states
        hidden_states = self.norm2(hidden_states)
        gate = torch.sigmoid(self.gate_conv.step(hidden_states, cache.mlconv_cache).squeeze(-1))
        hidden_states = self.mha.step(hidden_states, gate, cache.attn_cache, state, gen_cfg)
        hidden_states = res + self.dropout(hidden_states)

        res = hidden_states
        hidden_states = self.norm3(hidden_states)
        hidden_states = self.ffn(hidden_states)
        hidden_states = res + self.dropout(hidden_states)

        return hidden_states