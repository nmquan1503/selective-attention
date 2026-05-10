import torch
from typing import Tuple

def check_ndim(
    tensor: torch.Tensor | None,
    ndim: int,
    name: str | None = None,
    optional: bool = False
):
    if tensor is None:
        if not optional:
            message = "" if name else f"[{name}] "
            message += f"Expected tensor with ndim {ndim}, but got None"
            raise ValueError(message)
        return

    if tensor.ndim != ndim:
        prefix = f"[{name}] " if name else ""
        raise ValueError(
            f"{prefix}Dim mismatch: expected ndim={ndim}, got ndim={tensor.ndim}, shape={tuple(tensor.shape)}"
        )

def check_shape(
    tensor: torch.Tensor | None, 
    shape: Tuple[int], 
    name: str | None = None, 
    optional: bool = False
):
    if tensor is None:
        if not optional:
            message = "" if name else  f"[{name}] "
            message += f"Expected tensor with shape {shape}, but got None"
            raise ValueError(message)
    elif tensor.shape != shape:
        message = "" if name else  f"[{name}] "
        message += f"Shape mismatch: expected {shape}, got {tuple(tensor.shape)}"
        raise ValueError(message)

def to_contiguous(tensor: torch.Tensor | None):
    if tensor is None:
        return tensor
    if not tensor.is_contiguous():
        return tensor.contiguous()
    return tensor