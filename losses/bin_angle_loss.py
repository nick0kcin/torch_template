from torch.nn import Module
from torch.nn.functional import  max_pool2d
import torch
from torch.nn.functional import normalize, tanh, cross_entropy, l1_loss, log_softmax


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
            # labels = torch.fmod((gt[self.key][objects[:, 0], objects[:, 1], :] / self.bin_size + 0.5).long(), 90 / self.bin_size)
            # gt_ = torch.fmod(gt[self.key][objects[:, 0], objects[:, 1], :], 90)
            gt_ = gt[self.key][objects[:, 0], objects[:, 1], :]
            labels =(gt_ / self.bin_size).long().clamp(0, 90 / self.bin_size - 1)
            # labels_floor = torch.fmod((gt[self.key][objects[:, 0], objects[:, 1], :] / self.bin_size).long(),
            #                     90 / self.bin_size)
            # dense_gt = torch.zeros(predicted.shape).to(predict.device)
            # dense_gt[torch.arange(0,labels.shape[0]), labels[:,0]] =\
            #     torch.fmod((labels[:,0].float() + 1.5) * self.bin_size - gt[self.key][objects[:, 0], objects[:, 1], 0]+90,90)/self.bin_size
            # dense_gt[torch.arange(0,labels.shape[0]), torch.fmod(labels[:,0]+1, 90/self.bin_size).long()] =\
            # torch.fmod(90-(labels[:,0].float()+0.5)*self.bin_size + gt[self.key][objects[:, 0], objects[:, 1],0], 90) / self.bin_size
            loss_value = cross_entropy(predicted, labels[:, 0])
            # assert  (dense_gt.sum()-dense_gt.shape[0])<1e-03 and (dense_gt<0).sum()==0
            # assert ((dense_gt*torch.linspace(self.bin_size / 2,
            #                                  90 - self.bin_size / 2,
            #                                  90 // self.bin_size).to(predict.device)).sum(1) - gt[self.key][objects[:, 0], objects[:, 1],0]).sum(1)<1e-03
            # loss_value =(-dense_gt*log_softmax(predicted, dim=1)).sum(1).mean()
            # l1_loss_ = torch.min((predicted.argmax(dim=1).float() * self.bin_size - gt[self.key][objects[:, 0], objects[:, 1], 0]).abs(),
            #                      90 - (predicted.argmax(dim=1).float() * self.bin_size - gt[self.key][objects[:, 0], objects[:, 1], 0]).abs()).mean()
        else:
            loss_value = objects.sum().float()
            # loss_value =  predict.abs().mean()
        return loss_value