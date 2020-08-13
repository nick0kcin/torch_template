
from torch.nn import Module
from torch.nn.functional import  max_pool2d
import torch
from torch.nn.functional import normalize, tanh, cross_entropy, l1_loss, log_softmax, binary_cross_entropy_with_logits, relu



class Hinge(Module):
    def __init__(self, pos_weights):
        super(Hinge, self).__init__()
        self.pos_weight = pos_weights

    def forward(self, pred, gt):
        weights = 1 + gt["y"] * (self.pos_weight - 1)
        return (relu(1 - pred["y"] * (2 * gt["y"] - 1)) ** 2 * weights + 0.001 * pred["y"] ** 2).mean()