from torch.nn import Module
import torch
from torch.nn.functional import binary_cross_entropy_with_logits

class LabelSmoothing(Module):
    def __init__(self, smoothing=0.1):
        super(LabelSmoothing, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, x, target):
        if self.training:
            x = x["y"].float()
            target = target["y"].float()

            loss = (self.confidence * binary_cross_entropy_with_logits(x, target) + self.smoothing *
            binary_cross_entropy_with_logits(x, torch.ones_like(target) / 10))

            return loss
        else:
            weights = 13 * (target > 0.5) + (target < 0.5)
            return torch.nn.functional.cross_entropy(x, target, weights=weights)