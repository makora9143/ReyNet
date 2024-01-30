from typing import List

import torch
from torch import Tensor
from torch import nn
from torch.nn.functional import one_hot

from ..pooling import Maxpool
from ...utils import cyclic_perm_index, swap_positions
from .maron import SkipConnection2D


class SkipConnection1D(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()

        self.conv = nn.Conv1d(in_features, out_features, 1, 1, 0)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x1, x2):
        output = torch.cat([x1, x2], dim=1)
        output = self.conv(output)
        output = self.act(output)
        return output


class MaxConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, a, b, c):
        super().__init__()
        self.pooling = Maxpool(dim=-1, keepdim=True)
        self.conv = nn.Conv1d(in_channels, out_channels, 1, 1, 0)

    def forward(self, x):
        xm = self.pooling(x)
        return self.conv(x - xm)


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
            activation: nn.Module = nn.ReLU(inplace=True),
            skip_connection: bool = False
    ) -> None:

        super().__init__()
        self.in_features = in_features
        self.out_features = channel_layers[-1]
        self.channel_layers = channel_layers
        self.set_layers = set_layers
        self.skip_connection = skip_connection

        phi = [nn.Linear(1, set_layers[0]), activation]
        for i in range(1, len(set_layers)):
            phi += [nn.Linear(set_layers[i - 1], set_layers[i]), activation]
        phi += [nn.Linear(set_layers[-1], 1)]
        self.phi = nn.Sequential(*phi)

        if skip_connection:
            self.skip = SkipConnection1D(in_features * 2, in_features)

        fc = [MaxConv1d(in_features, channel_layers[0], 1, 1, 0), activation]
        for i in range(1, len(channel_layers)):
            fc += [
                MaxConv1d(channel_layers[i - 1], channel_layers[i], 1, 1, 0),
                activation
            ]
        self.channel_fc = nn.Sequential(*fc)

    def forward_old(self, x: Tensor):
        B, _, N = x.shape
        g_tensor = one_hot(torch.tensor(cyclic_perm_index(list(range(N)))),
                           num_classes=N).to(x)  # N x N x N

        h = x.reshape(B, self.in_features, 1, N, 1)  # B x C x 1 x N x 1

        h = g_tensor.matmul(h)  # B x C x N x N x 1
        h = self.phi(h[:, :, :, :2, 0])  # B x C x N x 1
        output = h.new_zeros(B, self.in_features, N, N)
        output[:, :, :, :1] = h
        output = g_tensor.transpose(-1, -2).matmul(output.reshape(B, self.in_features, N, N, 1))  # B x C x N x N x 1
        output = output.mean(2).squeeze(-1)  # B x C x N もしかしてsum(2)?
        if self.skip_connection:
            output = self.skip(x, output)
        output = self.channel_fc(output)
        return output

    def forward(self, x: Tensor):
        B, C, N = x.shape
        h = x.reshape(B, C, N, 1)
        h = self.phi(h)  # B x C x N x 1 -> B x C x N x H -> B x C x N x 1
        output = sum_ginv_x_1to1(h, N) / N  # B x C x N x 1 -> B x C x N
        if self.skip_connection:
            output = self.skip(x, output)
        output = self.channel_fc(output)  # B x C x N -> B x D x N
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
            skip_connection: bool = False,
    ) -> None:

        super().__init__()
        self.in_features = in_features
        self.out_features = channel_layers[-1] if channel_layers is not None else None
        self.set_layers = set_layers
        self.channel_layers = channel_layers
        self.skip_connection = skip_connection

        phi = [nn.Linear(2 * 2, set_layers[0]), nn.ReLU(inplace=True)]
        for i in range(1, len(set_layers)):
            phi += [nn.Linear(set_layers[i - 1], set_layers[i]), nn.ReLU(inplace=True)]
        phi += [nn.Linear(set_layers[-1], 2)]
        self.phi = nn.Sequential(*phi)

        if skip_connection:
            self.skip = SkipConnection2D(self.in_features * 2, self.in_features)

        if self.channel_layers is not None:
            fc = [nn.Conv2d(in_features, channel_layers[0], 1, 1, 0), nn.ReLU(inplace=True)]
            for i in range(1, len(channel_layers)):
                fc += [
                    nn.Conv2d(channel_layers[i - 1], channel_layers[i], 1, 1, 0),
                    nn.ReLU(inplace=True)
                ]
            self.channel_fc = nn.Sequential(*fc)

    def forward(self, x: Tensor, shortcut: bool = False):
        if shortcut:
            return self.shortcut_forward(x)
        return self.normal_forward(x)

    def shortcut_forward(self, x: Tensor) -> Tensor:
        B = x.size(0)
        x = x[:, :, :2, :2]  # B x C x N x N -> B x C x 2 x 2
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
        B, C, N, _ = x.shape

        coeff = (torch.ones(N, N).fill_diagonal_(0) +
                 torch.zeros(N, N).fill_diagonal_(1 / (N - 1))).reshape(1, 1, N, N).to(x)

        h = g_x_2to2(x, N)  # B x C x N x N -> B x C x N(N-1) x 2 x 2
        h = self.phi(h.reshape(B, C, N * (N - 1), -1))  # B x C x n! x 4 -> B x C x n! x 2
        output = sum_ginv_x_2to2(h, N)
        output = output * coeff  # B x C x N x N
        if self.skip_connection:
            output = self.skip(x, output)
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
        h = h[:, :, :, :2, :2]  # B x C x n! x 2 x 2
        h = self.phi(h.reshape(B, self.in_features, g_size, -1))  # B x C x n! x 4 -> B x C x n! x 2
        output = h.new_zeros(B, self.in_features, g_size, N, N)
        output[:, :, :, 0, :2] = h
        output = g_tensor.transpose(-1, -2).matmul(output).matmul(g_tensor)  # B x C x n! x n x n
        output = output.sum(2) * coeff.repeat(B, self.in_features, 1, 1)
        if self.residual:
            output += x
        h = self.channel_fc(output)  # B x D x N x N
        return output


def g_x_2to2(x, N):
    """ :math:`\mathcal{O}(N)` version
        :math:`\phi(x) = g x`
        x = mapping1(n, x)  # B x C x n x n -> B x C x n(n-1) x 2 x 2
    """
    B, C = x.shape[:2]
    x = x.repeat(1, 1, 2, 2)
    diag = x.diagonal(dim1=-2, dim2=-1)
    diag1 = diag[..., :N]
    result = []
    for i in range(1, N):
        diag2 = x.diagonal(offset=i, dim1=-2, dim2=-1)[..., :N]
        diag3 = x.diagonal(offset=-i, dim1=-2, dim2=-1)[..., :N]
        diag4 = diag[..., i:N + i]
        tmp = torch.stack([diag1, diag2, diag3, diag4], -1)
        result.append(tmp)
    return torch.cat(result, -2).reshape(B, C, -1, 2, 2)


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
           torch.tensor(cyclic_perm_index(list(range(N)))[1:]).flatten(
           )] = x[:, :, :, 1]

    return output + torch.diag_embed(
        x[..., 0].reshape(B, C, N - 1, N).sum(2))


def g_x_1to1(x, N):
    """
        :math:`\phi(x) = g x`
        B x C x N -> B x C x N! x 2
        B x C x N -> B x C x N x 2
    """
    x = torch.cat([x, x[:, :, :1]], -1)
    return torch.stack([x[:, :, i:i + 2] for i in range(N)], 2)


def sum_ginv_x_1to1(x, N):
    """
        :math:`\phi(x) = g x`
        B x C x N x 1 -> B x C x N
    """
    return x.squeeze(-1)
