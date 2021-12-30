from typing import List
from itertools import product

import torch
from torch import Tensor
from torch import nn
from torch.nn.functional import one_hot

from ...utils import cyclic_perm_index, swap_positions


class ReyEquiv1to1(nn.Module):
    """Reynolds Equivariant Layer for Rank-1

    Args:
        nn (int): The dimension of Input features
    """
    def __init__(
            self,
            in_features: int,
            set_layers: List[int] = [128],
            channel_layers: List[int] = [128],
            activation: nn.Module = nn.ReLU()
    ) -> None:

        super().__init__()
        self.in_features = in_features
        self.out_features = channel_layers[-1]
        self.channel_layers = channel_layers
        self.set_layers = set_layers

        phi = [nn.Linear(2, set_layers[0]), activation]
        for i in range(1, len(set_layers)):
            phi += [nn.Linear(set_layers[i - 1], set_layers[i]), activation]
        phi += [nn.Linear(set_layers[-1], 1)]
        self.phi = nn.Sequential(*phi)

        fc = [nn.Conv1d(in_features, channel_layers[0], 1, 1, 0), activation]
        for i in range(1, len(channel_layers)):
            fc += [
                nn.Conv1d(channel_layers[i - 1], channel_layers[i], 1, 1, 0),
                activation
            ]
        self.channel_fc = nn.Sequential(*fc)

    def forward_old(self, x: Tensor):
        B, _, N = x.shape
        g_tensor = one_hot(torch.tensor(cyclic_perm_index(list(range(N)))),
                           num_classes=N).to(x)  # N x N x N

        x = x.reshape(B, self.in_features, 1, N, 1)  # B x C x 1 x N x 1

        x = g_tensor.matmul(x)  # B x C x N x N x 1
        x = self.phi(x[:, :, :, :2, 0])  # B x C x N x 1
        output = x.new_zeros(B, self.in_features, N, N)
        output[:, :, :, :1] = x
        output = g_tensor.transpose(-1, -2).matmul(output.reshape(B, self.in_features, N, N, 1))  # B x C x N x N x 1
        output = output.mean(2).squeeze(-1)  # B x C x N もしかしてsum(2)?
        output = self.channel_fc(output)
        return output

    def forward_nonresidual(self, x: Tensor):
        B, _, N = x.shape

        x = g_x_1to1(x, N)  # B x C x N -> B x C x N x 2
        x = self.phi(x)  # B x C x N x 2 -> B x C x N x 1
        output = sum_ginv_x_1to1(x, N) / N  # B x C x N x 1 -> B x C x N
        output = self.channel_fc(output)
        return output

    def shortcut(self, x):
        return x

    def forward(self, x: Tensor):
        B, _, N = x.shape
        h = g_x_1to1(x, N)  # B x C x N -> B x C x N x 2
        h = self.phi(h)  # B x C x N x 2 -> B x C x N x 1
        output = sum_ginv_x_1to1(h, N) / N  # B x C x N x 1 -> B x C x N
        output += x
        output = self.channel_fc(output)
        return output


class ReyEquiv2to2(nn.Module):
    r"""Reynolds Equivariant Layer 2to2

    Args:
        set_layers: the list of the number of set-wise layer units
        channel_layers: the list of the number of channel-wise layer units

    Shape:
        - Inputs: :math:`(N, C, S, S)`
        - Outputs: :math:`(N, D, S, S)`

    Examples::
        >>> m = ReyEquiv2to2(3, set_layers=[128, 128], channel_layers=[128, 128])
        >>> x = torch.randn(16, 3, 5, 5)
        >>> output = m(x)
        >>> print(output.size())
        torch.Size([16, 128, 5, 5])
    """
    def __init__(
            self,
            in_features: int,
            set_layers: List[int] = [128],
            channel_layers: List[int] = [128],
            num_corner: int = 2,
            residual: bool = False,
    ) -> None:

        super().__init__()
        self.in_features = in_features
        self.out_features = channel_layers[-1] if channel_layers is not None else None
        self.set_layers = set_layers
        self.channel_layers = channel_layers
        self.num_corner = num_corner
        self.residual = residual

        phi = [nn.Linear(num_corner * num_corner, set_layers[0]), nn.ReLU()]
        for i in range(1, len(set_layers)):
            phi += [nn.Linear(set_layers[i - 1], set_layers[i]), nn.ReLU()]
        phi += [nn.Linear(set_layers[-1], 2)]
        self.phi = nn.Sequential(*phi)

        if self.channel_layers is not None:
            fc = [nn.Conv2d(in_features, channel_layers[0], 1, 1, 0), nn.ReLU()]
            for i in range(1, len(channel_layers)):
                fc += [
                    nn.Conv2d(channel_layers[i - 1], channel_layers[i], 1, 1, 0),
                    nn.ReLU()
                ]
            self.channel_fc = nn.Sequential(*fc)

    def forward(self, x: Tensor, shortcut: bool = False):
        if shortcut:
            return self.shortcut_forward(x)
        return self.normal_forward(x)

    def shortcut_forward(self, x: Tensor) -> Tensor:
        B = x.size(0)
        x = x[:, :, :self.num_corner, :self.num_corner]  # B x C x N x N -> B x C x 2 x 2
        x = self.phi(x.reshape(B, self.in_features, -1))  # B x C x 4 -> B x C x 2
        x = self.channel_fc(x.unsqueeze(-1)).squeeze(-1)
        return x

    def normal_forward(self, x: Tensor) -> Tensor:
        """
            # Set direction operation
            x = g_x_2to2(n, x)  # B x C x n x n -> B x C x n(n-1) x 2 x 2
            x = self.phi(x.reshape(B, self.in_features, g_size, -1))  # B x C x n(n-1) x 4 -> B x C x n(n-1) x 2
            x = sum_ginv_x_2to2(n, x)  # B x C x n(n-1) x 2 -> B x C x n x n

            # Channel direction operation
            x = self.channel_fc(x)  # B x D x N x N
            return output
        """
        B, _, N, _ = x.shape

        coeff = (torch.ones(N, N).fill_diagonal_(0) +
                 torch.zeros(N, N).fill_diagonal_(1 / (N - 1))).reshape(1, 1, N, N).to(x)

        h = g_x_2to2_old(x, N)  # B x C x N x N -> B x C x N(N-1) x 2 x 2
        h = self.phi(h.reshape(B, self.in_features, N * (N - 1), -1))  # B x C x n! x 4 -> B x C x n! x 2
        output = sum_ginv_x_2to2(h, N)
        output = output * coeff.repeat(B, self.in_features, 1, 1)  # B x C x N x N
        if self.residual:
            output += x
        if self.channel_layers is not None:
            output = self.channel_fc(output)  # B x D x N x N
        return output

    def normal_forward_old(self, x: Tensor) -> Tensor:
        B, _, N, _ = x.shape

        g_tensor = one_hot(torch.tensor([cyclic_perm_index(swap_positions(list(range(N)), 1, i))
                                         for i in range(1, N)]).reshape(-1, N),
                           num_classes=N).unsqueeze(0).float().to(x)  # 1 x n! x n x n
        coeff = (torch.ones(N, N).fill_diagonal_(0) +
                 torch.zeros(N, N).fill_diagonal_(1 / (N - 1))).reshape(1, 1, N, N).to(x)

        g_size = g_tensor.size(1)

        h = x.unsqueeze(2)  # B x C x n x n -> B x C x 1 x n x n
        h = g_tensor.matmul(h).matmul(g_tensor.transpose(-1, -2))  # B x C x n! x n x n
        h = h[:, :, :, :self.num_corner, :self.num_corner]  # B x C x n! x 2 x 2
        h = self.phi(h.reshape(B, self.in_features, g_size, -1))  # B x C x n! x 4 -> B x C x n! x 2
        output = h.new_zeros(B, self.in_features, g_size, N, N)
        output[:, :, :, 0, :2] = h
        output = g_tensor.transpose(-1, -2).matmul(output).matmul(g_tensor)  # B x C x n! x n x n
        output = output.sum(2) * coeff.repeat(B, self.in_features, 1, 1)
        if self.residual:
            output += x
        h = self.channel_fc(output)  # B x D x N x N
        return output


def g_x_2to2_old(x, N):
    """
        :math:`\phi(x) = g x`
        x = mapping1(n, x)  # B x C x n x n -> B x C x n(n-1) x 2 x 2
    """
    x = x.repeat(1, 1, 2, 2)
    return torch.stack([
        x[:, :, j:i + j + 2:i + 1, j:i + j + 2:i + 1]
        for i, j in product(range(N - 1), range(N))
    ], 2)


def g_x_2to2(x, N):
    """
        :math:`\phi(x) = g x`
        x = mapping1(n, x)  # B x C x N x N -> B x C x N! x 2 x 2
        x = mapping1(n, x)  # B x C x N x N -> B x C x N(N-1) x 2 x 2
    """
    result = []
    for j in range(N):
        result += [x[:, :, 0:i + 2:i + 1, 0:i + 2:i + 1] for i in range(N - 1)]
        x = x.roll((-1, -1), dims=(-2, -1))
    return torch.stack(result, 2)


def sum_ginv_x_2to2(x, N):
    """
        :math:`\rho(x) = \sum g^{-1} x`
        x = mapping2(n, x)  # B x C x n(n-1) x 2 -> B x C x n x n
        peak memory: 1244.94 MiB, increment: 4.80 MiB
    """
    B, C = x.shape[:2]
    output = x.new_zeros(B, C, N, N)
    output[:, :,
           torch.tensor(list(range(N)) * (N - 1)),
           torch.tensor(cyclic_perm_index(list(range(N)))[1:]).ravel(
           )] = x[:, :, :, 1]

    return output + torch.diag_embed(
        torch.stack([x[:, :, i::N, 0] for i in range(N)], -1).sum(2))


def g_x_1to1(x, N):
    """
        :math:`\phi(x) = g x`
        B x C x N -> B x C x N! x 2
        B x C x N -> B x C x N x 2
    """
    x = torch.cat([x[:, :], x[:, :, :1]], -1)
    return torch.stack([x[:, :, i:i + 2] for i in range(N)], 2)


def sum_ginv_x_1to1(x, N):
    """
        :math:`\phi(x) = g x`
        B x C x N x 1 -> B x C x N
    """
    return x.squeeze(-1)
