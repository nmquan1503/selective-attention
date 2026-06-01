import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple

from ..ops import SSDScanFn, SSUFn
from ..inference import SSMCache

class SSM(nn.Module):
    def __init__(
        self,
        layer_idx: int,
        model_dim: int = 512,
        state_dim: int = 128,
        conv_kernel_size: int = 4,
        head_dim: int = 64,
        num_groups: int = 1,
        expansion_factor: int = 2,
        chunk_size: int = 256,
        delta_limit: Tuple[float, float] = (0.0, float("inf")),
        A_init_range: Tuple[int, int] = (1, 16),
        delta_init_limit: Tuple[float, float] = (0.001, 0.1),
        delta_init_floor: float = 1e-4,
        dropout_rate: float = 0.15,
        device="cuda"
    ):
        super().__init__()
        self.layer_idx = layer_idx

        self.model_dim = model_dim
        self.state_dim = state_dim
        self.inner_dim = expansion_factor * model_dim
        assert self.inner_dim % head_dim == 0
        self.conv_kernel_size = conv_kernel_size
        self.head_dim = head_dim
        self.num_heads = self.inner_dim // head_dim
        self.num_groups = num_groups
        assert self.num_heads % num_groups == 0
        self.chunk_size = chunk_size
        self.delta_limit = delta_limit

        self.in_proj = nn.Linear(self.model_dim, 2 * self.inner_dim + 2 * self.num_groups * self.state_dim + self.num_heads)

        conv_dim = self.inner_dim + 2 * self.num_groups * self.state_dim
        self.conv = nn.Conv1d(
            in_channels=conv_dim,
            out_channels=conv_dim,
            kernel_size=conv_kernel_size,
            groups=conv_dim,
            padding=conv_kernel_size - 1
        )

        delta_init = torch.exp(
            torch.rand(self.num_heads, device=device, dtype=torch.float32) 
            * (math.log(delta_init_limit[1]) - math.log(delta_init_limit[0]))
            + math.log(delta_init_limit[0])
        )
        delta_init = torch.clamp(delta_init, min=delta_init_floor)
        inv_delta_init = delta_init + torch.log(-torch.expm1(-delta_init))
        self.delta_bias = nn.Parameter(inv_delta_init)
        self.delta_bias._no_weight_decay = True
    
        A = torch.empty(self.num_heads, dtype=torch.float32, device=device).uniform_(*A_init_range)
        A_log = torch.log(A)
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True

        self.D = nn.Parameter(torch.ones(self.inner_dim, device=device))
        self.D._no_weight_decay = True

        self.out_proj = nn.Linear(self.inner_dim, self.model_dim)
    
        self.dropout = nn.Dropout(dropout_rate)

    def forward(
        self, 
        hidden_states: torch.Tensor, 
        lengths: torch.Tensor | None = None,
        ssm_hiddens: torch.Tensor | None = None, 
        cache: SSMCache | None = None
    ):
        """
        Args:
            hidden_states: (batch_size, seq_len, model_dim)
            lengths: (batch_size,)
            ssm_hiddens: (batch_size, inner_dim, state_dim)
        
        Returns: 
            hidden_states: (batch_size, seq_len, model_dim)
            last_ssm_hiddens: (batch_size, inner_dim, state_dim)
        """
        batch_size, seq_len, _ = hidden_states.shape
        device = hidden_states.device
        is_infer = cache is not None

        hBC, gate_logits, delta_raw = torch.split(
            self.in_proj(hidden_states),
            [
                self.inner_dim + 2 * self.num_groups * self.state_dim,
                self.inner_dim,
                self.num_heads
            ],
            dim=-1
        )

        A = -torch.exp(self.A_log)

        hBC = hBC.transpose(1, 2)

        if is_infer:
            cache_size = self.conv_kernel_size
            cache.build_conv_ctx(hBC, lengths, cache_size)
        
        hBC = F.silu(hBC, inplace=is_infer)
        hBC = self.conv(hBC).transpose(1, 2)
        hBC = hBC[:, :seq_len]

        hidden_states, B, C = torch.split(
            hBC,
            [
                self.inner_dim,
                self.num_groups * self.state_dim,
                self.num_groups * self.state_dim
            ],
            dim=-1
        )

        ssm_residual = hidden_states
        
        hidden_states = hidden_states.view(batch_size, seq_len, self.num_heads, self.head_dim)
        B = B.view(batch_size, seq_len, self.num_groups, self.state_dim)
        C = C.view(batch_size, seq_len, self.num_groups, self.state_dim)

        hidden_states, last_ssm_hiddens = SSDScanFn.apply(
            hidden_states, A, B, C, delta_raw, self.delta_bias, ssm_hiddens, lengths,
            self.chunk_size, True, self.delta_limit, True
        )

        if is_infer:
            cache.h = last_ssm_hiddens
        
        hidden_states = hidden_states.view(batch_size, seq_len, -1)
        if is_infer:
            torch.addcmul(hidden_states, ssm_residual, self.D, value=1.0, out=hidden_states)
            gate_logits.sigmoid_()
            hidden_states.mul_(gate_logits)
        else:
            hidden_states = hidden_states + self.D * ssm_residual
            hidden_states = hidden_states * F.sigmoid(gate_logits)

        hidden_states = self.out_proj(hidden_states)

        return hidden_states, last_ssm_hiddens

    def step(self, hidden_states: torch.Tensor, cache: SSMCache):
        """
        Args: (batch_size, model_dim)
        Returns: (batch_size, model_dim)
        """
        batch_size = hidden_states.shape[0]

        hBC, gate_logits, delta_raw = torch.split(
            self.in_proj(hidden_states),
            [
                self.inner_dim + 2 * self.num_groups * self.state_dim,
                self.inner_dim,
                self.num_heads
            ],
            dim=-1
        )

        cache.update(hBC)
        hBC = cache.conv_ctx
        hBC = F.silu(hBC)
        hBC = (hBC * self.conv.weight.squeeze(1)).sum(dim=-1) + self.conv.bias
        hidden_states, B, C = torch.split(
            hBC,
            [
                self.inner_dim,
                self.num_groups * self.state_dim,
                self.num_groups * self.state_dim
            ],
            dim=-1
        )

        A = -torch.exp(self.A_log)

        ssm_residual = hidden_states

        hidden_states = hidden_states.view(batch_size, self.num_heads, self.head_dim)
        B = B.view(batch_size, self.num_groups, self.state_dim)
        C = C.view(batch_size, self.num_groups, self.state_dim)

        hidden_states, cache.h = SSUFn.forward(
            hidden_states, A, B, C, delta_raw, self.delta_bias, cache.h
        )

        hidden_states = hidden_states.view(batch_size, -1)
        torch.addcmul(hidden_states, ssm_residual, self.D, value=1.0, out=hidden_states)
        gate_logits.sigmoid_()
        hidden_states.mul_(gate_logits)
        hidden_states = self.out_proj(hidden_states)
        
        return hidden_states