from torch.nn import Module


def identity(x):
    return x


class Acc(Module):
    def __init__(self, pos_weights=0.5):
        super(Acc, self).__init__()
        self.pos_weight = pos_weights

    def forward(self, pred, gt):
        return ((pred["label"] > self.pos_weight) == gt["label"]).float()
