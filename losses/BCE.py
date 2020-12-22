from torch.nn import Module
from torch.nn.functional import binary_cross_entropy_with_logits


def identity(x):
    return x


class BCE(Module):
    def __init__(self, pos_weights=None):
        super(BCE, self).__init__()
        self.pos_weight = pos_weights
        self.epoch = -1

    def forward(self, pred, gt):
        weights = 1 + gt["label"] * (self.pos_weight - 1) if self.pos_weight else None
        return binary_cross_entropy_with_logits(pred["label"], gt["label"], weights)