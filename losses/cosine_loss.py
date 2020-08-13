from torch.nn import Module
from torch.nn.functional import  max_pool2d
import torch
from torch.nn.functional import normalize, tanh


def identity(x):
    return x


class CosineLoss(Module):
    def __init__(self, key, activation=None):
        super(CosineLoss, self).__init__()
        self.key = key
        self.activation = globals()[activation] if activation else identity

    def forward(self, pred, gt):
        predict = pred[self.key]
        objects = (gt["pos"][:, :, 0] >= 0).nonzero()
        if objects.nelement():
            positions = gt["pos"][objects[:, 0], objects[:, 1], :]
            predicted = normalize(self.activation(predict[objects[:, 0], :, positions[:, 1], positions[:, 0]]))
            labels = gt[self.key][objects[:, 0], objects[:, 1], :]
            loss_value = (1 - (predicted * labels).sum(dim=1)).mean() #((predicted - labels) ** 2).mean().sqrt() #(1 - (predicted * labels).sum(dim=1)).mean()
        else:
            loss_value = objects.sum().float()
            # loss_value =  predict.abs().mean()
        return loss_value