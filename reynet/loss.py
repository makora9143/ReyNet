from torch import Tensor
from torch.nn import Module
from torch.nn.functional import mse_loss


class CornerMSELoss(Module):
    def forward(self,
                input: Tensor,
                target: Tensor,
                corner: bool = True) -> Tensor:
        if corner:
            target = target[:, :, 0, :2]
        loss = mse_loss(input, target)
        return loss
