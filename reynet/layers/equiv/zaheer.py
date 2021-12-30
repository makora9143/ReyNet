from torch import nn

from ..pooling import Maxpool


class PermEqui1_max(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.pooling = Maxpool(dim=1, keepdim=True)
        self.gamma = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        """
        B x N x C -> B x N x D
        """
        xm = self.pooling(x)
        return self.gamma(x - xm)
