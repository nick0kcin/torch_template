from torch.nn import Module
from torch.nn.functional import  max_pool2d
import torch
from torch.nn.functional import normalize, tanh, cross_entropy, l1_loss, log_softmax, binary_cross_entropy_with_logits


def identity(x):
    return x


class BCE(Module):
    def __init__(self, pos_weights):
        super(BCE, self).__init__()
        self.pos_weight = pos_weights

    def forward(self, pred, gt):
        weights = 1 + gt["y"] * (self.pos_weight - 1)
        return binary_cross_entropy_with_logits(pred["y"], gt["y"], weights)