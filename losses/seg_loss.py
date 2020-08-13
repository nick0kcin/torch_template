from torch.nn import Module
from torch.nn.functional import binary_cross_entropy_with_logits
import torch


class SegLoss(Module):

    def __init__(self, bce_weight=0.05, dice_smooth=1.):
        super(SegLoss, self).__init__()
        self.bce_weight = bce_weight
        self.dice_smooth = dice_smooth

    def dice(self, pred, gt):
        pred = pred.contiguous()
        gt = gt.contiguous()
        intersection = (pred * gt).sum(dim=2).sum(dim=2)
        union = gt.sum(dim=2).sum(dim=2) + pred.sum(dim=2).sum(dim=2)
        dice_value = (2 * intersection + self.dice_smooth) / (union + self.dice_smooth)
        return 1 - dice_value.mean()

    def forward(self, predict, gt):
        pred = predict[:, :, :gt.shape[2], :gt.shape[3]]
        bce = binary_cross_entropy_with_logits(pred, gt)
        c_map = pred.sigmoid()
        loss = self.bce_weight * bce + (1 - self.bce_weight) * self.dice(c_map, gt)
        return loss


def get_seg_loss(bce_weight=0.05, dice_smooth=1.):
    def get_loss():
        return SegLoss(bce_weight, dice_smooth)
    return get_loss
