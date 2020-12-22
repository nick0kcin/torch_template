from torch.nn import Module
from torch.nn.functional import l1_loss


class L1Loss(Module):
    def __init__(self):
        super(L1Loss, self).__init__()

    def forward(self, predict, gt):
        pred = predict[:, :, :gt.shape[2], :gt.shape[3]]
        return l1_loss(pred,  gt, reduction="sum")


def get_l1_loss():
    return L1Loss()
