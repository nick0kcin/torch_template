from torch.nn import Module
from torch.nn.functional import l1_loss


def identity(x):
    return x


class L1SparseLoss(Module):
    def __init__(self, key, activation=None):
        super(L1SparseLoss, self).__init__()
        self.key = key
        self.activation = globals()[activation] if activation else identity

    def forward(self, pred, gt):
        predict = pred[self.key]
        if "pos" in gt:
            objects = (gt["pos"][:, :, 0] >= 0).nonzero()
            if objects.nelement():
                positions = gt["pos"][objects[:, 0], objects[:, 1], :]
                predicted = self.activation(predict[objects[:, 0], :, positions[:, 1], positions[:, 0]])
                labels = gt[self.key][objects[:, 0], objects[:, 1], :]
                loss_value = l1_loss(predicted, labels)
            else:
                loss_value = objects.sum().float()
                # loss_value =  predict.abs().mean()
            return loss_value
        else:
            return l1_loss(predict, gt[self.key])


def get_l1_sparse_loss():
    return L1SparseLoss()
