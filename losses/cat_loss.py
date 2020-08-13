from torch.nn import Module
from torch.nn.functional import  max_pool2d
import torch
from torch.nn.functional import normalize, tanh, cross_entropy, l1_loss, log_softmax


def identity(x):
    return x


class CatLoss(Module):
    def __init__(self, key):
        super(CatLoss, self).__init__()
        self.key = key

    def forward(self, pred, gt):
        predict = pred[self.key]
        objects = (gt["pos"][:, :, 0] >= 0).nonzero()
        if objects.nelement():
            positions = gt["pos"][objects[:, 0], objects[:, 1], :]
            predicted = predict[objects[:, 0], :, positions[:, 1], positions[:, 0]]
            gt_ = gt[self.key][objects[:, 0], objects[:, 1], :]
            loss_value = cross_entropy(predicted, gt_[:, 0])
        else:
            loss_value = objects.sum().float()
            # loss_value =  predict.abs().mean()
        return loss_value