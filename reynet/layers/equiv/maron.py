from typing import List
import math

import torch
from torch import Tensor
from torch import nn


############################################
# Operator
############################################

class Ops2to2(nn.Module):
    """Operation Basis for 2 x 2 (15types)
    """
    def __init__(
            self,
            normalization: str = 'inf',
            normalization_val: float = 1.0
    ) -> None:
        super().__init__()

        self.normalization = normalization
        self.normalization_val = normalization_val

    def forward(self, input: Tensor) -> List[Tensor]:
        dim = input.size(-1)
        diag_part = input.diagonal(dim1=-1, dim2=-2)
        sum_diag_part = diag_part.sum(2, keepdims=True)
        sum_of_rows = input.sum(3)
        sum_of_cols = input.sum(2)
        sum_all = sum_of_rows.sum(2)

        op1 = torch.diag_embed(diag_part, dim1=-1, dim2=-2)

        op2 = torch.diag_embed(sum_diag_part.repeat(1, 1, dim), dim1=-1, dim2=-2)

        op3 = torch.diag_embed(sum_of_rows, dim1=-1, dim2=-2)

        op4 = torch.diag_embed(sum_of_cols, dim1=-1, dim2=-2)

        op5 = torch.diag_embed(sum_all.unsqueeze(2).repeat(1, 1, dim), dim1=-1, dim2=-2)

        op6 = sum_of_cols.unsqueeze(3).repeat(1, 1, 1, dim)

        op7 = sum_of_rows.unsqueeze(3).repeat(1, 1, 1, dim)

        op8 = sum_of_cols.unsqueeze(2).repeat(1, 1, dim, 1)

        op9 = sum_of_rows.unsqueeze(2).repeat(1, 1, dim, 1)

        op10 = input

        op11 = input.permute(0, 1, 3, 2)

        op12 = diag_part.unsqueeze(3).repeat(1, 1, 1, dim)

        op13 = diag_part.unsqueeze(2).repeat(1, 1, dim, 1)

        op14 = sum_diag_part.unsqueeze(3).repeat(1, 1, dim, dim)

        op15 = sum_all.unsqueeze(2).unsqueeze(3).repeat(1, 1, dim, dim)

        if self.normalization is not None:
            float_dim = float(dim)
            if self.normalization == 'inf':
                op2 = op2.div(float_dim)
                op3 = op3.div(float_dim)
                op4 = op4.div(float_dim)
                op5 = op5.div(float_dim ** 2)
                op6 = op6.div(float_dim)
                op7 = op7.div(float_dim)
                op8 = op8.div(float_dim)
                op9 = op9.div(float_dim)
                op14 = op14.div(float_dim)
                op15 = op15.div(float_dim ** 2)

        return [op1, op2, op3, op4, op5, op6, op7, op8, op9, op10, op11, op12, op13, op14, op15]


class Ops1to1(nn.Module):
    """Operation Basis for 1 x 1 (2types)
    """
    def __init__(
            self,
            normalization: str = 'inf',
            normalization_val=1.0
    ) -> None:
        super().__init__()

        self.normalization = normalization
        self.normalization_val = normalization_val

    def forward(self, input: Tensor) -> List[Tensor]:
        dim = input.size(-1)
        sum_all = input.sum(2, keepdims=True)  # N x D x 1
        op1 = input
        op2 = sum_all.repeat(1, 1, dim)

        if self.normalization is not None:
            float_dim = float(dim)
            if self.normalization == 'inf':
                op2 = op2.div(float_dim)

        return [op1, op2]


#############################################
# Linear Layer
#############################################

class MaronLinear2to2(nn.Module):
    """Maron Linear Layer

    (B x C x N x N x 1) -> fixed(1 x 15) -> (B x N x N x 15) * C x 15 x D -> B x D x N x N x 1

    Args:

    """
    def __init__(
            self,
            in_features: int,
            out_features: int,
            basis_dimension: int = 15
    ) -> None:
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.basis_dimension = basis_dimension

        self.coeffs = nn.Parameter(
            torch.empty(in_features, out_features, self.basis_dimension).normal_() * math.sqrt(2. / float(in_features + out_features))
        )
        self.diag_bias = nn.Parameter(torch.zeros(1, out_features, 1, 1))
        self.all_bias = nn.Parameter(torch.zeros(1, out_features, 1, 1))

    def forward(self, x: Tensor) -> Tensor:
        set_size = x.size(-1)
        output = torch.einsum('dsb,ndbij->nsij', self.coeffs, x)

        mat_diag_bias = torch.eye(set_size).unsqueeze(0).unsqueeze(0).to(x) * self.diag_bias

        return output + self.all_bias + mat_diag_bias

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, basis_dimension={self.basis_dimension}"


class MaronLinear1to1(nn.Module):
    def __init__(
            self,
            in_features: int,
            out_features: int,
            basis_dimension: int
    ) -> None:
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.basis_dimension = basis_dimension

        self.coeffs = nn.Parameter(
            torch.empty(in_features, out_features, basis_dimension).normal_() * math.sqrt(2. / float(in_features + out_features))
        )
        self.bias = nn.Parameter(torch.zeros(1, out_features, 1))

    def forward(self, x: Tensor) -> Tensor:
        output = torch.einsum('dsb,ndbi->nsi', self.coeffs, x)
        output = output + self.bias
        return output

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, basis_dimension={self.basis_dimension}"


##############################################
# Equivariant Layer
##############################################

class MaronEquiv2to2Layer(nn.Module):
    def __init__(
            self,
            in_features: int,
            out_features: int,
            normalization: str = 'inf',
            normalization_val: float = 1.0
    ) -> None:
        super().__init__()

        self.basis_dimension = 15
        self.in_features = in_features
        self.out_features = out_features

        self.ops_2_to_2 = Ops2to2(normalization=normalization, normalization_val=normalization_val)
        self.fc = MaronLinear2to2(in_features, out_features, self.basis_dimension)

    def forward(self, x: Tensor) -> Tensor:
        ops_out = torch.stack(self.ops_2_to_2(x), 2)  # B x C x 15 x N x N
        output = self.fc(ops_out)
        return output


class MaronEquiv1to1Layer(nn.Module):
    def __init__(
            self,
            in_features: int,
            out_features: int,
            normalization: str = 'inf',
            normalization_val: float = 1.0
    ) -> None:
        super().__init__()

        self.basis_dimension = 2
        self.in_features = in_features
        self.out_features = out_features

        self.ops_1_to_1 = Ops1to1(normalization=normalization, normalization_val=normalization_val)
        self.fc = MaronLinear1to1(in_features, out_features, self.basis_dimension)

    def forward(self, x: Tensor) -> Tensor:
        ops_out = torch.stack(self.ops_1_to_1(x), 2)
        output = self.fc(ops_out)
        return output

####################################################
# Equivariant Component
####################################################


class MaronEquivNet2to2(nn.Module):
    """Maron Equivariant Layer for 2x2

    Args:
        in_features: C
        layers: [.., D]

    Inputs:
        B x C x N x N
    Outputs:
        B x D x N x N
    """
    def __init__(self, in_features: int, layers: List[int] = [16]) -> None:
        super().__init__()
        self.in_features = in_features
        self.layers = layers

        net: List[nn.Module] = [
            MaronEquiv2to2Layer(in_features, layers[0]),
            nn.ReLU(),
        ]

        for i in range(1, len(layers)):
            net += [MaronEquiv2to2Layer(layers[i - 1], layers[i]), nn.ReLU()]

        self.net = nn.Sequential(*net)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class MaronEquivNet1to1(nn.Module):
    """Maron Equivariant Layer for 1x1

    Args:
        in_features: C
        layers: [.., D]

    Inputs:
        B x C x N
    Outputs:
        B x D x N
    """
    def __init__(self, in_features: int, layers: List[int] = [16]) -> None:
        super().__init__()
        self.in_features = in_features
        self.hidden_layers = layers

        net: List[nn.Module] = [
            MaronEquiv1to1Layer(in_features, layers[0]),
            nn.ReLU(),
        ]

        for i in range(1, len(layers)):
            net += [MaronEquiv1to1Layer(layers[i - 1], layers[i]), nn.ReLU()]

        self.net = nn.Sequential(*net)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class SkipConnection2D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, 1, 1, 0)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x1, x2):
        output = torch.cat([x1, x2], dim=1)
        output = self.conv(output)
        output = self.act(output)
        return output


class Equi2to2onlyMLP(nn.Module):
    def __init__(self, in_channels, out_channels, depth_of_mlp):
        super().__init__()

        net = [nn.Conv2d(in_channels, out_channels, 1, 1, 0), nn.ReLU(inplace=True)]
        for i in range(1, depth_of_mlp):
            net.append(nn.Conv2d(out_channels, out_channels, 1, 1, 0))
            net.append(nn.ReLU(inplace=True))
        self.net = nn.Sequential(*net)

    def forward(self, x):
        return self.net(x)


class RegularBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.mlp1 = Equi2to2onlyMLP(in_channels, out_channels, depth_of_mlp=3)
        self.mlp2 = Equi2to2onlyMLP(in_channels, out_channels, depth_of_mlp=3)
        self.skip = SkipConnection2D(in_channels + out_channels, out_channels)

    def forward(self, x):
        block = x
        output1 = self.mlp1(block)
        output2 = self.mlp2(block)
        output = torch.matmul(output1, output2)
        output = self.skip(x, output)
        return output
