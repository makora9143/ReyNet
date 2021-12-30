import torch
from torch import Tensor
from torch import nn


class Sumpool(nn.Module):
    def __init__(self, dim: int = -1, keepdim: bool = False) -> None:
        super().__init__()
        self.dim = dim
        self.keepdim = keepdim

    def forward(self, x: Tensor) -> Tensor:
        return x.sum(self.dim, keepdim=self.keepdim)


class Maxpool(nn.Module):
    def __init__(self, dim: int = -1, keepdim: bool = False):
        super().__init__()
        self.dim = dim
        self.keepdim = keepdim

    def forward(self, x: Tensor) -> Tensor:
        x, _ = x.max(self.dim, keepdim=self.keepdim)
        return x


class DiagOffdiagMaxpool(nn.Module):
    """Diagonal and Off-diagonal Max pooling

    Permutation Invariant Layer.

    Shape:
        - Inputs: :math:`(N, C, S, S)`
        - Outputs: :math:`(N, 2C)`

    Examples:
        >>> m = DiagOffdiagMaxpool()
        >>> x = torch.randn(12, 3, 4, 4)
        >>> output = m(x)
        >>> output.size()
        torch.Size([12, 6])
    """
    def forward(self, x: Tensor) -> Tensor:
        max_diag = x.diagonal(dim1=-1, dim2=-2).max(2)[0]
        max_val = max_diag.max()
        min_val = x.mul(-1.).max()
        val = (max_val + min_val).abs()
        min_mat = torch.diag_embed(x[0][0].diagonal(dim1=-1,
                                                    dim2=-2).mul(0).add(val),
                                   dim1=-1,
                                   dim2=-2).unsqueeze(0).unsqueeze(0)
        max_offdiag = x.sub(min_mat).max(3)[0].max(2)[0]
        return torch.cat([max_diag, max_offdiag], 1)


class DiagOffdiagSumpool(nn.Module):
    """Diagonal and Off-diagonal Sum pooling

    Permutation Invariant Layer.

    Shape:
        - Inputs: :math:`(N, C, S, S)`
        - Outputs: :math:`(N, 2C)`

    Examples:
        >>> m = DiagOffdiagSumpool()
        >>> x = torch.randn(12, 3, 4, 4)
        >>> output = m(x)
        >>> output.size()
        torch.Size([12, 6])
    """
    def forward(self, x: Tensor) -> Tensor:
        sum_diag = x.diagonal(dim1=-1, dim2=-2).sum(-1)
        sum_offdiag = x.sum([-1, -2]) - sum_diag
        return torch.cat([sum_diag, sum_offdiag], 1)
