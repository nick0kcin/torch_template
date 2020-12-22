import torch
from torch.nn import Module


class FocalLoss(Module):

    def __init__(self, key, sigmoid=True, a=2, b=4, thr=0.0):
        super(FocalLoss, self).__init__()
        self.a = a
        self.b = b
        self.key = key
        self.sigmoid = sigmoid
        self.thr = thr

    def forward(self, pred, gt):
        gt = gt[self.key].sigmoid() if gt[self.key].min() < 0 else gt[self.key]
        pred = pred[self.key] if isinstance(pred, dict) else pred
        if gt.max() == 1:
            pos_inds = (gt == 1.0).float()
            neg_inds = gt.lt(1).float()
        else:
            maxs = torch.nn.functional.max_pool2d(gt, kernel_size=3, stride=1, padding=0)
            pos_inds = torch.zeros_like(gt)
            maxs = (maxs == gt[:, :, 1:-1, 1:-1]) & (maxs > 0.2)
            pos_inds[:, :, 1:-1, 1:-1] = maxs
            pos_inds *= gt
            neg_inds = (torch.ones_like(gt) - pos_inds) * (1 - gt)

        neg_weights = torch.pow(1 - gt, self.b).float()
        loss = 0

        c_map = pred.sigmoid().clamp(1e-05, 1 - 1e-05) if self.sigmoid else pred.clamp(1e-05, 1 - 1e-05)

        pos_loss = torch.log(c_map) * torch.pow(1 - c_map, self.a) * pos_inds  # * sizes
        neg_loss = torch.log(1 - c_map) * torch.pow(c_map, self.a) * neg_weights * neg_inds

        num_pos = pos_inds.float().sum(1).sum(1).sum(1)
        pos_loss = pos_loss.sum(1).sum(1).sum(1)
        neg_loss = neg_loss.sum(1).sum(1).sum(1)

        if num_pos.sum() == 0:
            loss = loss - neg_loss
        else:
            loss = loss - (pos_loss + neg_loss) / num_pos.clamp(min=1)
        if loss.requires_grad:
            loss_value = (loss * (loss > self.thr).float()).sum() / (loss > self.thr).sum()
        else:
            loss_value = (loss * (loss > self.thr).float()).sum() / (loss > self.thr).sum()

        return loss_value


def get_focal_loss(sigmoid=True, a=2, b=4):
    def get_loss():
        return FocalLoss(sigmoid, a, b)

    return get_loss
