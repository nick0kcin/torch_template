import torch
from torch.nn import Module


class PairLoss(Module):
    def __init__(self,):
        super(PairLoss, self).__init__()

    def forward(self, pred, gt):
        pos = pred["y"][gt["y"] > 0.5]
        neg = pred["y"][gt["y"] < 0.5]
        diff = neg.view(-1, neg.shape[0]) - pos.view(pos.shape[0], -1)
        return (torch.exp(diff).sum() - neg.shape[0]) / 2
