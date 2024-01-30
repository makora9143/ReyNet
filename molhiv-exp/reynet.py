from functools import partial

from torch import nn
from torch import Tensor
import torch

from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder
from torch_geometric.utils import to_dense_adj, to_dense_batch


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


class DataEmbedding(nn.Module):
    def __init__(self, emb_dim: int, mol: bool = False):
        '''
        Args:
            emb_dim (int) : node/edge embedding dimensionality
            mol (bool):

        Shape:
            - Outputs: B x (2 x emb_dim) x S x S
        '''
        super().__init__()
        self.mol = mol

        if self.mol:
            self.node_encoder = AtomEncoder(emb_dim)
            self.edge_encoder = BondEncoder(emb_dim)
        else:
            self.node_encoder = nn.Embedding(1, emb_dim)
            self.edge_encoder = nn.Linear(7, emb_dim)

    def reset_parameters(self):
        if self.mol:
            for emb in self.edge_encoder.bond_embedding_list:
                nn.init.xavier_uniform_(emb.weight.data)
            for emb in self.node_encoder.atom_embedding_list:
                nn.init.xavier_uniform_(emb.weight.data)
        else:
            self.edge_encoder.reset_parameters()
            nn.init.xavier_uniform_(self.node_encoder.weight.data)

    def forward(self, batch_data):
        x, edge_index, edge_attr, batch = batch_data.x, batch_data.edge_index, batch_data.edge_attr, batch_data.batch

        h = self.node_encoder(x)
        edge_embedding = self.edge_encoder(edge_attr)

        return torch.cat([
            torch.diag_embed(to_dense_batch(h, batch)[0].transpose(-1, -2)),
            to_dense_adj(edge_index, batch, edge_embedding).permute(0, 3, 1, 2)
        ], 1)


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
            set_emb: int,
            hidden_channels: int,
            num_setlayer: int = 1,
            num_convlayer: int = 1,
            skip: bool = False,
            device = torch.device('cpu'),
    ) -> None:
        super().__init__()

        self.skip = skip
        self.device = device

        convs = [
            nn.Linear(2 * 2, set_emb),
            # nn.BatchNorm1d(set_emb),
            nn.ReLU(inplace=True)
        ]
        for i in range(2):
            convs += [
                nn.Linear(set_emb, set_emb),
                # nn.BatchNorm1d(set_emb),
                nn.ReLU(inplace=True)
            ]
        convs += [nn.Linear(set_emb, 2)]
        self.set_convs = nn.Sequential(*convs)

        if skip:
            self.skip_path = SkipConnection2D(4 * hidden_channels, 2 * hidden_channels)
        else:
            self.skip_path = lambda x, output: output

        # fc = [nn.Conv2d(2 * hidden_channels, 2 * hidden_channels, 1, 1, 0),
        #       nn.BatchNorm2d(2 * hidden_channels), nn.ReLU(inplace=True)]
        # for i in range(num_convlayer):
        #     fc += [nn.Conv2d(2 * hidden_channels, 2 * hidden_channels, 1, 1, 0), nn.BatchNorm2d(2 * hidden_channels), nn.ReLU(inplace=True)]
        # fc += [nn.Conv2d(2 * hidden_channels, hidden_channels, 1, 1, 0)]

        fc = [
            nn.Conv2d(2 * hidden_channels, 128, 1, 1, 0),
            # nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 1, 1, 0),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 1, 1, 0),
            # nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, hidden_channels, 1, 1, 0),
        ]

        self.channel_fc = nn.Sequential(*fc)

    def reset_parameters(self):
        for c in self.set_convs.children():
            if hasattr(c, 'reset_parameters'):
                c.reset_parameters()

    def set_device(self, device):
        self.device = device

    # def forward(self, x: Tensor, omit: bool = False):
    #     if omit:
    #         return self.omit_forward(x)
    #     return self.regular_forward(x)
    def forward(self, x: Tensor):
        return self.regular_forward(x)

    def omit_forward(self, x: Tensor) -> Tensor:
        B, C = x.size(0), x.size(1)
        x = x[..., :2, :2]  # slice left corner: B x C x N x N -> B x C x 2 x 2
        x = self.set_convs(x.reshape(B * C, -1)).reshape(B, C, -1, 1)  # conv set: B x C x 4 -> B x C x 2 x 1
        x = self.channel_fc(x).squeeze(-1)  # conv features: B x C x 2 x 1 -> B x D x 2
        return x

    def regular_forward(self, x: Tensor) -> Tensor:
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
                 torch.zeros(N, N).fill_diagonal_(1 / (N - 1))).reshape(1, 1, N, N).to(self.device)

        h = g_x_2to2(x, N)  # B x C x N x N -> B x C x N(N-1) x 2 x 2
        h = self.set_convs(h.reshape(B * C * N * (N - 1), -1)).reshape(B, C, N * (N - 1), -1)  # B x C x n! x 4 -> B x C x n! x 2
        output = sum_ginv_x_2to2(h, N)
        output = output * coeff  # B x C x N x N
        output = self.skip_path(x, output)
        output = self.channel_fc(output)  # B x D x N x N
        return output


class SetChBNReyEquiv2to2(ReyEquiv2to2):
    def __init__(
            self,
            set_emb: int,
            hidden_channels: int,
            num_setlayer: int = 1,
            num_convlayer: int = 1,
            skip: bool = False,
            device=torch.device('cpu'),
    ) -> None:
        super().__init__(
            set_emb,
            hidden_channels,
            num_setlayer,
            num_convlayer,
            skip,
            device,
        )

        self.skip = skip
        self.device = device

        convs = [
            nn.Linear(2 * 2, set_emb),
            nn.BatchNorm1d(set_emb),
            nn.ReLU(inplace=True)
        ]
        for i in range(2):
            convs += [
                nn.Linear(set_emb, set_emb),
                nn.BatchNorm1d(set_emb),
                nn.ReLU(inplace=True)
            ]
        convs += [nn.Linear(set_emb, 2)]
        self.set_convs = nn.Sequential(*convs)

        if skip:
            self.skip_path = SkipConnection2D(4 * hidden_channels, 2 * hidden_channels)
        else:
            self.skip_path = lambda x, output: output

        fc = [
            nn.Conv2d(2 * hidden_channels, 128, 1, 1, 0),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 1, 1, 0),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 1, 1, 0),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, hidden_channels, 1, 1, 0),
        ]

        self.channel_fc = nn.Sequential(*fc)


class SetBNReyEquiv2to2(ReyEquiv2to2):
    def __init__(
            self,
            set_emb: int,
            hidden_channels: int,
            num_setlayer: int = 1,
            num_convlayer: int = 1,
            skip: bool = False,
            device=torch.device('cpu'),
    ) -> None:
        super().__init__(
            set_emb,
            hidden_channels,
            num_setlayer,
            num_convlayer,
            skip,
            device,
        )

        self.skip = skip
        self.device = device

        convs = [
            nn.Linear(2 * 2, set_emb),
            nn.BatchNorm1d(set_emb),
            nn.ReLU(inplace=True)
        ]
        for i in range(2):
            convs += [
                nn.Linear(set_emb, set_emb),
                nn.BatchNorm1d(set_emb),
                nn.ReLU(inplace=True)
            ]
        convs += [nn.Linear(set_emb, 2)]
        self.set_convs = nn.Sequential(*convs)

        if skip:
            self.skip_path = SkipConnection2D(4 * hidden_channels, 2 * hidden_channels)
        else:
            self.skip_path = lambda x, output: output

        fc = [
            nn.Conv2d(2 * hidden_channels, 128, 1, 1, 0),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 1, 1, 0),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 1, 1, 0),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, hidden_channels, 1, 1, 0),
        ]

        self.channel_fc = nn.Sequential(*fc)


class ChBNReyEquiv2to2(ReyEquiv2to2):
    def __init__(
            self,
            set_emb: int,
            hidden_channels: int,
            num_setlayer: int = 1,
            num_convlayer: int = 1,
            skip: bool = False,
            device=torch.device('cpu'),
    ) -> None:
        super().__init__(
            set_emb,
            hidden_channels,
            num_setlayer,
            num_convlayer,
            skip,
            device,
        )
        self.skip = skip
        self.device = device

        convs = [
            nn.Linear(2 * 2, set_emb),
            nn.ReLU(inplace=True)
        ]
        for i in range(2):
            convs += [
                nn.Linear(set_emb, set_emb),
                nn.ReLU(inplace=True)
            ]
        convs += [nn.Linear(set_emb, 2)]
        self.set_convs = nn.Sequential(*convs)

        if skip:
            self.skip_path = SkipConnection2D(4 * hidden_channels, 2 * hidden_channels)
        else:
            self.skip_path = lambda x, output: output

        fc = [
            nn.Conv2d(2 * hidden_channels, 128, 1, 1, 0),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 1, 1, 0),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 1, 1, 0),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, hidden_channels, 1, 1, 0),
        ]

        self.channel_fc = nn.Sequential(*fc)


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


class ReyNet(nn.Module):

    def __init__(self,
                 hidden_channels: int,
                 out_channels: int,
                 num_layer: int,
                 dropout: float,
                 skip: bool = False,
                 mol: bool = False,
                 setbn: int = 0,
                 chbn: int = 0,
                 setchbn: int = 0,
                 lastbn: int = 0,
                 device=torch.device('cpu')) -> None:
        super().__init__()
        self.mol = mol
        self.device = device
        self.input_encoder = DataEmbedding(hidden_channels, mol)

        if setbn:
            reyequiv = SetBNReyEquiv2to2
        elif chbn:
            reyequiv = ChBNReyEquiv2to2
        elif setchbn:
            reyequiv = SetChBNReyEquiv2to2
        else:
            reyequiv = ReyEquiv2to2

        self.rey_equiv = nn.DataParallel(reyequiv(16, hidden_channels, 1, 1, skip=skip))
        self.pooling = DiagOffdiagSumpool()

        if lastbn:
            self.mlp = nn.Sequential(
                nn.Linear(2 * hidden_channels, hidden_channels),
                nn.BatchNorm1d(hidden_channels),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout),
                nn.Linear(hidden_channels, out_channels))
        else:
            self.mlp = nn.Sequential(
                nn.Linear(2 * hidden_channels, hidden_channels),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout),
                nn.Linear(hidden_channels, out_channels))

    def reset_parameters(self):
        self.input_encoder.reset_parameters()
        self.rey_equiv.module.reset_parameters()
        for c in self.mlp.children():
            if hasattr(c, 'reset_parameters'):
                c.reset_parameters()

    def forward(self, batch_data) -> Tensor:
        input = self.input_encoder(batch_data)
        h = self.rey_equiv(input)
        # h = self.rey_equiv(input, omit=False)
        h = self.pooling(h)
        output = self.mlp(h)
        return output

    def set_device(self, device):
        self.device = device
        self.rey_equiv.module.set_device(device)


def reorder_from_idx(idx, input_list):
    """Re-order by index idx

    input [a, b, c, d, e] -> if idx = 2 then [c, d, e, a, b]
    """
    len_a = len(input_list)
    return [(i + idx) % len_a for i in input_list]


def cyclic_perm_index(input_list):
    """return cyclic-index
    """
    return [partial(reorder_from_idx, i)(input_list) for i in range(len(input_list))]


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
           )] = x[..., 1]

    return output + torch.diag_embed(
        x[..., 0].reshape(B, C, N - 1, N).sum(2))
