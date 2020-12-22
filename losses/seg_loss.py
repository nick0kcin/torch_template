from torch.nn import Module
from torch.nn.functional import binary_cross_entropy_with_logits, binary_cross_entropy
import torch
import numpy as np

__all__ = ["SegLoss"]

class SegLoss(Module):

    def __init__(self, bce_weight=0.2, dice_weight=None, dice_smooth=1., gamma=0.3, thr=None, inv=False):
        super(SegLoss, self).__init__()
        if bce_weight:
            self.bce_weight = np.stack((np.ones((len(bce_weight),)), bce_weight))
            self.bce_weight /= self.bce_weight.sum(0, keepdims=True)
        if dice_weight:
            self.dice_weight = np.stack((np.ones((len(dice_weight),)), dice_weight))
            self.dice_weight /= self.dice_weight.sum(0, keepdims=True)
        else:
            self.dice_weight = None
        self.dice_smooth = dice_smooth
        self.thr = thr
        self.inv = inv
        self.gamma = gamma
        self.epoch = -1

    def dice(self, pred, gt, a = 0.5):
        pred = pred if self.thr is None else (pred > self.thr).float()
        gt = gt
        intersection = (pred * gt).sum(dim=-1).sum(dim=-1)
        # union = intersection + a * (pred * (1 - gt)).sum(dim=-1).sum(dim=-1) + (1 - a)*(gt* (1- pred)).sum(dim=-1).sum(dim=-1)
        all_union = gt.sum(dim=-1).sum(dim=-1) + pred.sum(dim=-1).sum(dim=-1)
        # gt.sum(dim=-1).sum(dim=-1) + pred.sum(dim=-1).sum(dim=-1)
        # print(gt.sum(dim=-1).sum(dim=-1))
        # print(pred.sum(dim=-1).sum(dim=-1))
        # print(intersection)
        # print((2 * intersection + self.dice_smooth) / (union + self.dice_smooth))
        # assert (intersection > aunion).sum()==0
        # assert ((intersection + self.dice_smooth) / (union + self.dice_smooth) - 2*(intersection + self.dice_smooth) / (all_union + self.dice_smooth)).abs().sum() < 1e-08
        dice_value = (intersection + self.dice_smooth) / (all_union + self.dice_smooth)
        return 1 - dice_value if self.inv else dice_value

    def forward(self, predict, gt):
        pred = predict["mask"][:, :, :gt["mask"].shape[2], :gt["mask"].shape[3]]
        bces = []
        dices = []
        kls = []
        for i in range(pred.shape[1]):
            w = (gt["mask"][:, i, : :] < 0.5) * self.bce_weight[0, i] + (gt["mask"][:, i, : :] > 0.5) * self.bce_weight[1, i]
            assert ((pred[:, i, :, :].clamp(1e-06, 1 - 1e-06) < 0) | (pred[:, i, :, :].clamp(1e-06, 1 - 1e-06) > 1)).sum() == 0
            assert ((gt["mask"][:, i, :, :] < 0) | (
                        gt["mask"][:, i, :, :] > 1)).sum() == 0
            bce = binary_cross_entropy_with_logits(
            pred[:, i, :, :], gt["mask"][:, i, :, :], weight=w, reduce=False) + 0.1 * (gt["mask"][:, i, :, :].sum() > 0) * binary_cross_entropy_with_logits(
            pred[:, i, :, :], 0.5*torch.ones_like(gt["mask"][:, i, :, :]), reduce=False)

            if "weight" in gt:
                w = gt["weight"][:, i]
            elif self.dice_weight is None:
                w = 1
            else:
                w = ((gt["mask"].detach()>0.5).sum(-1).sum(-1)[:, i] > 0) * self.dice_weight[1, i] + ((gt["mask"].detach() >0.5).sum(-1).sum(-1)[:, i] == 0) * self.dice_weight[0, i]
            wdice = torch.clamp(self.dice(pred[:, i, :, :].sigmoid(), gt["mask"][:, i, : :]), min=1e-4)
            dice = torch.pow(-torch.log(wdice), self.gamma)
            assert (wdice <= 0).sum() ==0
            assert (wdice > 1).sum() ==0,wdice
            assert torch.isnan(dice).sum()==0
            bces.append(bce.sum(-1).sum(-1) / (bce.shape[-1] * bce.shape[-2]))
            dices.append(dice * w)
            if "kl" in predict:
                kls.append(predict["kl"][:, i, :, :].sum(-1).sum(-1) / (bce.shape[-1] * bce.shape[-2]))

        bce = torch.stack(bces, dim=1)
        dice = torch.stack(dices, dim=1)
        if "kl" in predict:
            kl = torch.stack(kls, dim=1)
        # pt = torch.exp(bce)

        # compute the loss
        # focal_loss = -((1 - pt) ** 1) * bce
        c_map = pred
        if not c_map.requires_grad:
            w = 0.9 * (gt["mask"].sum(-1).sum(-1) > 5) + 0.1
        # loss = self.bce_weight * focal_loss.view(focal_loss.shape[0], focal_loss.shape[1], -1).mean(-1) + (1 - self.bce_weight)  * self.dice(c_map, gt["mask"])
        loss = (1 - 0.2) * dice + 0.2 * bce
        if "kl" in predict and self.training and self.epoch > 2:
            kl_weight = (self.epoch - 2) * 0.1 if self.epoch < 12 else 1.0
            loss += kl * kl_weight
        return loss # * (~((gt["label"]==0) & (predict["label"].detach() < 0))).float()


def get_seg_loss(bce_weight=0.05, dice_smooth=1.):
    def get_loss():
        return SegLoss(bce_weight, dice_smooth)
    return get_loss
