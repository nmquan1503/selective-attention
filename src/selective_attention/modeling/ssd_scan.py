import torch
from typing import Tuple

from selective_attention.utils.tensor_utils import  check_ndim, check_shape, to_contiguous
from selective_attention.triton_kernels.ssd_scan.forward import ssd_scan_forward
from selective_attention.triton_kernels.ssd_scan.backward import ssd_scan_backward

class SSDScanFn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, 
        u: torch.Tensor,
        A: torch.Tensor, 
        B: torch.Tensor, 
        C: torch.Tensor, 
        delta_raw: torch.Tensor,
        delta_bias: torch.Tensor | None = None,
        h_init: torch.Tensor | None = None,
        length: torch.Tensor | None = None,
        chunk_size: int = 256,
        use_delta_softplus: bool = True,
        delta_limit: Tuple[float, float] = (0.0, float("inf")),
        is_causal: bool = True,
    ):
        """
        Args:
            u: (batch_size, seq_len, num_heads, head_dim)
            A: (num_heads,)
            B, C: (batch_size, seq_len, num_groups, state_dim)
            delta_raw: (batch_size, seq_len, num_heads)
            delta_bias: (num_heads,)
            h_init: (batch_size, num_heads, head_dim, state_dim)
            length: (batch_size,)
        
        Returns:
            y: (batch_size, seq_len, num_heads, head_dim)
            h_last: (batch_size, num_heads, head_dim, state_dim)
        """
        check_ndim(u, 4, name="u")
        check_ndim(A, 1, name="A")
        check_ndim(B, 4, name="B")
        check_ndim(C, 4, name="C")
        check_ndim(delta_raw, 3, name="delta_raw")
        check_ndim(delta_bias, 1, name="delta_bias", optional=True)
        check_ndim(h_init, 4, name="h_init", optional=True)
        check_ndim(length, 1, name="length", optional=True)

        batch_size, seq_len, num_heads, head_dim = u.shape
        num_groups, state_dim = B.shape[2], B.shape[3]

        check_shape(A, (num_heads,), name="A")
        check_shape(B, (batch_size, seq_len, num_groups, state_dim), name="B")
        check_shape(C, (batch_size, seq_len, num_groups, state_dim), name="C")
        check_shape(delta_raw, (batch_size, seq_len, num_heads), name="delta_raw")
        check_shape(delta_bias, (num_heads,), name="delta_bias", optional=True)
        check_shape(h_init, (batch_size, num_heads, head_dim, state_dim), name="h_init", optional=True)
        check_shape(length, (batch_size,), name="length", optional=True)

        u = to_contiguous(u)
        A = to_contiguous(A)
        B = to_contiguous(B)
        C = to_contiguous(C)
        delta_raw = to_contiguous(delta_raw)
        delta_bias = to_contiguous(delta_bias)
        h_init = to_contiguous(h_init)
        length = to_contiguous(length)

        if length is None:
            length = torch.full((batch_size,), seq_len, dtype=torch.long, device="cuda")

        y, h_last = ssd_scan_forward(
            u, A, B, C, delta_raw, delta_bias, h_init, length,
            chunk_size, use_delta_softplus, delta_limit, is_causal
        )

        ctx.save_for_backward(u, A, B, C, delta_raw, delta_bias, h_init, length)
        ctx.chunk_size = chunk_size
        ctx.use_delta_softplus = use_delta_softplus
        ctx.delta_limit = delta_limit
        ctx.is_causal = is_causal

        return y, h_last

    @staticmethod
    def backward(ctx, y_grad, h_last_grad):
        u, A, B, C, delta_raw, delta_bias, h_init, length = ctx.saved_tensors
        chunk_size = ctx.chunk_size
        use_delta_softplus = ctx.use_delta_softplus
        delta_limit = ctx.delta_limit
        is_causal = ctx.is_causal

        u_grad, A_grad, B_grad, C_grad, delta_raw_grad, delta_bias_grad, h_init_grad = ssd_scan_backward(
            u, A, B, C, delta_raw, delta_bias, h_init, y_grad, h_last_grad, length,
            chunk_size, use_delta_softplus, delta_limit, is_causal
        )

        return u_grad, A_grad, B_grad, C_grad, delta_raw_grad, delta_bias_grad, h_init_grad, None, None, None, None, None

