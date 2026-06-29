import torch
import torch.nn as nn

class RMSNorm(nn.Module):

    def __init__(
        self, 
        dim: int, 
        num_groups: int | None = None,
        eps: float = 1e-6
    ):
        super().__init__()
        self.eps = eps
        self.num_groups = num_groups

        if num_groups is None:
            self.weight = nn.Parameter(torch.zeros(dim))
        else:
            self.weight = nn.Parameter(torch.zeros(num_groups, dim))

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
            if num_groups is None:
                x: (..., dim)
            else:
                x: (..., num_groups, dim)
        """
        output = self._norm(x.float())
        output = output * (1.0 + self.weight.float())
        return output.type_as(x)