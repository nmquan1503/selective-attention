import torch
import torch.nn as nn

class RMSNorm(nn.Module):

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))
    
    def _norm(self, x: torch.Tensor):
        """
        Args:
            x: (..., dim)

        Returns:
            (..., dim)
        """
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (batch_size, seq_len, dim)

        Returns:
            (batch_size, seq_len, dim)
        """
        output = self._norm(x.float())
        output = output * (1.0 + self.weight.float())
        return output.type_as(x)