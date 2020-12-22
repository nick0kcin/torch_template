import torch
from torch.nn import Module


class CorellationLoss(Module):
    def __init__(self, key):
        super(CorellationLoss, self).__init__()
        self.key = key

    def forward(self, predict, gt):
        predict = predict[self.key]
        gt = gt[self.key]
        assert gt.shape == predict.shape
        mean_pred = predict.sum(-2, keepdim=True).sum(-1, keepdim=True) / (predict.shape[-1] * predict.shape[-2])
        mean_gt = gt.sum(-2, keepdim=True).sum(-1, keepdim=True) / (gt.shape[-1] * gt.shape[-2])
        predict = predict - mean_pred
        gt = gt - mean_gt
        mean_gt = mean_gt[:, :, 0, 0] - mean_gt[:, :, 0, 0].mean(1, keepdim=True)
        mean_pred = mean_pred[:, :, 0, 0] - mean_pred[:, :, 0, 0].mean(1, keepdim=True)
        c1 = (predict * gt).sum(-1).sum(-1) / torch.sqrt(
            (predict ** 2).sum(-1).sum(-1) * (gt ** 2).sum(-1).sum(-1) + 1e-06)
        c2 = (mean_gt * mean_pred).sum(-1) / torch.sqrt((mean_pred ** 2).sum(-1) * (mean_gt ** 2).sum(-1) + 1e-06)
        assert -1 <= c1.min() and c1.max() <= 1 and -1 <= c2.min() and c2.max() <= 1
        l1 = torch.sqrt(1 - c1 * c1)
        l2 = torch.sqrt(1 - c2 * c2)
        return l1.mean() + l2.mean()


def get_l1_loss():
    return L1Loss()
