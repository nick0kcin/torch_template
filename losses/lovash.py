import torch
from torch.nn import Module


class Lovash(Module):

    def __init__(self):
        super(Lovash, self).__init__()
        self.max_diff = 0
        self.moving_avg = 0

    def lovasz_grad(self, gt_sorted):
        p = len(gt_sorted)
        gts = gt_sorted.sum()
        intersection = gts - gt_sorted.float().cumsum(0)
        union = gts + (1 - gt_sorted).float().cumsum(0)
        jaccard = 1. - intersection / union
        if p > 1:  # cover 1-pixel case
            jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
        return jaccard

    def lovash(self, pred, gt):
        sign_gt = 2 * gt - 1
        errors = (gt - pred).abs()
        err_sorted, perm = torch.sort(errors, dim=0, descending=True)
        perm = perm.data
        gt_sorted = gt[perm]
        grad = self.lovasz_grad(gt_sorted)
        loss = torch.dot(err_sorted, grad)
        return loss

    def forward(self, predict, gt):
        pred = predict["mask"][:, :, :gt["mask"].shape[2], :gt["mask"].shape[3]]
        loss = torch.zeros((1, pred.shape[1]), device=pred.device)
        for i in range(pred.shape[1]):
            loss[0, i] = self.lovash(pred[:, i, :, :].flatten(), gt["mask"][:, i, :, :].flatten())
            if gt["mask"][:, i, :, :].sum() > 1e-08:
                a = (pred > 0.5).float()
                lovash_iou = self.lovash(a[:, i, :, :].flatten(), gt["mask"][:, i, :, :].flatten())
                c = (a[:, i, :, :] * gt["mask"][:, i, :, :]).sum()
                cc = (pred[:, i, :, :] * gt["mask"][:, i, :, :]).sum()
                iou = 1.0 - c / (a[:, i, :, :].sum() + gt["mask"][:, i, :, :].sum() - c)
                iou_c = 1.0 - cc / (pred[:, i, :, :].sum() + gt["mask"][:, i, :, :].sum() - cc)
                self.max_diff = max(self.max_diff, (iou_c - loss[0, i]).abs().cpu().item())
                self.moving_avg = 0.99 * self.moving_avg + 0.01 * (iou_c - loss[0, i]).abs().cpu().item()
                assert (lovash_iou - iou) < 1e-02, f"{lovash_iou}, {iou}"
        return loss
