from torch.nn import Module
from torch.nn.functional import cross_entropy


def identity(x):
    return x


class BinAngleLoss(Module):
    def __init__(self, key, bin_size):
        super(BinAngleLoss, self).__init__()
        self.key = key
        self.bin_size = bin_size

    def forward(self, pred, gt):
        predict = pred[self.key]
        objects = (gt["pos"][:, :, 0] >= 0).nonzero()
        if objects.nelement():
            positions = gt["pos"][objects[:, 0], objects[:, 1], :]
            predicted = predict[objects[:, 0], :, positions[:, 1], positions[:, 0]]
            gt_ = gt[self.key][objects[:, 0], objects[:, 1], :]
            labels = (gt_ / self.bin_size).long().clamp(0, 90 / self.bin_size - 1)
            loss_value = cross_entropy(predicted, labels[:, 0])
        else:
            loss_value = objects.sum().float()
        return loss_value
