from torch.nn import Module
from torch.nn.functional import relu


class Hinge(Module):
    def __init__(self, pos_weights):
        super(Hinge, self).__init__()
        self.pos_weight = pos_weights

    def forward(self, pred, gt):
        weights = 1 + gt["y"] * (self.pos_weight - 1)
        return (relu(1 - pred["y"] * (2 * gt["y"] - 1)) ** 2 * weights + 0.001 * pred["y"] ** 2).mean()
