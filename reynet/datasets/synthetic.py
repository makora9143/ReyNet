import torch
from torch import Tensor
from torch.utils.data import Dataset


class SyntheticDataset(Dataset):
    __repr_indent = 4

    type: str = ""

    def __init__(self, fname: str, set_size: int = 40, train: bool = True, seed: int = None, size: int = 10000):
        self.set_size = set_size
        self.train = train
        self.size = size if train else 1000
        self.seed = seed
        if seed is not None:
            torch.manual_seed(seed)

        self.data = torch.empty(self.size, set_size, set_size).uniform_(0, 10)

        self.input_dim = 1
        self.output_dim = 1
        self.fname = fname

    def y(self, x) -> Tensor:
        raise NotImplementedError

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index):
        x = self.data[index]
        return x.unsqueeze(0), self.y(x).unsqueeze(0)

    def __repr__(self) -> str:
        head = "{} {} ({})".format(self.__class__.__name__, self.fname, "Train" if self.train else "Test")
        body = [
            f"Type: {self.type}",
            f"Number of datapoints: {self.__len__()}",
            f"Size of input set: {self.set_size}",
        ]
        body += self.extra_repr().splitlines()
        if self.seed is not None:
            body.append(f"Seed: {self.seed}")
        lines = [head] + [" " * self.__repr_indent + line for line in body]
        return '\n'.join(lines)

    def extra_repr(self) -> str:
        return ""


class EquivariantDataset(SyntheticDataset):
    r"""
    Symmetry Dataset.
        :math:`\frac{1}{2}(X + X^\top)`


    Power Dataset.
        :math:`x ^ 2`

    Diagonal Dataset.
        :math:`diag(diag(X))`

    Non-Diagonal Dataset.
        :math:`\{x_{ij} = 0\}_{i=j}`

    Inverse Dataset.
        :math:`X^{-1}`
    """

    type = 'equivariant'

    equiv_functions = {
        'symmetry': lambda x: 0.5 * (x + x.transpose(-1, -2)),
        'power': lambda x: x.pow(2),
        'diagonal': lambda x: torch.diag_embed(x.diagonal(dim1=-2, dim2=-1), dim1=-2, dim2=-1),
        'nondiagonal': lambda x: (torch.ones(x.size(-1), x.size(-1)) - torch.eye(x.size(-1))) * x,
        'inverse': lambda x: torch.inverse(x),
        'detinverse': lambda x: torch.inverse(x) * torch.det(x)
    }

    def __init__(self, fname: str, set_size: int = 40, train: bool = True, seed: int = None, size: int = 10000):
        super().__init__(fname, set_size, train, seed, size)
        self.y = self.equiv_functions[fname]


class InvariantDataset(SyntheticDataset):
    r"""
    Sum of power 2
        :math: \sum_{i,j} X_{i,j}^2

    Trace Dataset.
        :math:`tr(X)`

    Trace Dataset.
        :math:`det(X)`

    Product all elements of X.
        :math:`\prod_{i,j} x_{ij}`

    Sum of all elements
        :math:`\sum_{i,j} x_{ij}`

    Sum of Non-diagonal elements
        :math:`\sum_{i \neq j} x_{ij}`
    """
    type = 'invariant'

    inv_functions = {
        'powersum': lambda x: x.pow(2).sum(),
        'trace': lambda x: x.trace(),
        'determinant': lambda x: torch.det(x),
        'prod': lambda x: x.prod(),
        'sum': lambda x: x.sum(),
        'nondiagsum': lambda x: x.sum() - x.diagonal().sum()
    }

    def __init__(self, fname: str, set_size: int = 40, train: bool = True, seed: int = None, size: int = 10000):
        super().__init__(fname, set_size, train, seed, size)
        self.y = self.inv_functions[fname]
