from torch.nn import Module
from torch.nn.functional import  max_pool2d
import torch
from torch.nn.functional import normalize, tanh, cross_entropy, l1_loss, softmax


def identity(x):
    return x


class BinExpectationAngleLoss(Module):
    def __init__(self, key, predict_key, bin_size):
        super(BinExpectationAngleLoss, self).__init__()
        self.key = key
        self.predict_key = predict_key
        self.bin_size = bin_size

    def forward(self, pred, gt):
        predict = pred[self.predict_key]
        objects = (gt["pos"][:, :, 0] >= 0).nonzero()
        if objects.nelement():
            positions = gt["pos"][objects[:, 0], objects[:, 1], :]
            # gt_ = torch.fmod(gt[self.key][objects[:, 0], objects[:, 1], :], 90)
            gt_ = gt[self.key][objects[:, 0], objects[:, 1], :]
            labels = (gt_ / self.bin_size).long().clamp(0, 90 / self.bin_size - 1)
            predicted = predict[objects[:, 0], :, positions[:, 1], positions[:, 0]].tanh() * self.bin_size / 2
            expected = self.bin_size / 2 + labels.float() * self.bin_size - gt_
            assert (expected>self.bin_size/2).sum()==0 and (expected<-self.bin_size/2).sum()==0
            loss_value = l1_loss(expected, predicted)
        else:
            loss_value = objects.sum().float()
            # loss_value =  predict.abs().mean()
        return loss_value