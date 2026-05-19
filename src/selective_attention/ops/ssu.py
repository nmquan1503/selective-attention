import torch
import torch.nn as nn
from typing import Tuple

from ..triton_kernels import ssu_forward
from ..utils import check_ndim, check_shape, to_contiguous

class SSUFn(nn.Module):
    @staticmethod
    @torch.inference_mode()
    def forward(
        self, 
        u: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        delta_raw: torch.Tensor,
        delta_bias: torch.Tensor | None,
        h: torch.Tensor,
        use_delta_softplus: bool = True,
        delta_limit: Tuple = (0.0, float("inf"))
    ):
        """
        Args:
            u: (batch_size, num_heads, head_dim)
            A: (num_heads,)
            B, C: (batch_size, num_groups, state_dim)
            delta_raw: (batch_size, num_heads)
            delta_bias: (num_heads,)
            h: (batch_size, num_heads, head_dim, state_dim)
        
        Returns:
            y: (batch_size, num_heads, head_dim)
            h: (batch_size, num_heads, head_dim, state_dim)
        """
        check_ndim(u, 4, name="u")
        check_ndim(A, 1, name="A")
        check_ndim(B, 3, name="B")
        check_ndim(C, 3, name="C")
        check_ndim(delta_raw, 2, name="delta_raw")
        check_ndim(delta_bias, 1, name="delta_bias", optional=True)
        check_ndim(h, 4, name="h")

        batch_size, num_heads, head_dim = u.shape
        num_groups, state_dim = B.shape[1], B.shape[2]

        check_shape(A, (num_heads,), name="A")
        check_shape(B, (batch_size, num_groups, state_dim), name="B")
        check_shape(C, (batch_size, num_groups, state_dim), name="C")
        check_shape(delta_raw, (batch_size, num_heads), name="delta_raw")
        check_shape(delta_bias, (num_heads,), name="delta_bias", optional=True)
        check_shape(h, (batch_size, num_heads, head_dim, state_dim), name="h")

        u = to_contiguous(u)
        A = to_contiguous(A)
        B = to_contiguous(B)
        C = to_contiguous(C)
        delta_raw = to_contiguous(delta_raw)
        delta_bias = to_contiguous(delta_bias)
        h = to_contiguous(h)

        y, h = ssu_forward(u, A, B, C, delta_raw, delta_bias, h, use_delta_softplus, delta_limit)

        return y, h
