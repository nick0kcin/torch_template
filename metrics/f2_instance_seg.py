import numpy as np


def iou(gt, pred):
    i = ((gt * pred) > 0).sum().float()
    u = ((gt + pred) > 0).sum().float() + 1e-09
    return i / u


def f2(gt, pred, thresholds=tuple(np.arange(0.5, 1, 0.05))):
    if gt.sum() == 0:
        return float(pred.sum() < 1e-08)

    ious = []
    mp_idx_found = []
    for i in range(gt.shape[0]):
        mt = gt[i, :, :]
        for mp_idx in range(pred.shape[0]):
            if mp_idx not in mp_idx_found:
                cur_iou = iou(mt, pred[mp_idx, :, :])
                if cur_iou > 0.5:
                    ious.append(float(cur_iou))
                    mp_idx_found.append(mp_idx)
                    break
    f2_total = 0.0
    for thr in thresholds:
        tp = float(sum([iou_value > thr for iou_value in ious]))
        fn = gt.shape[0] - tp
        fp = pred.shape[0] - tp
        f2_total += (5. * tp) / (5. * tp + 4 * fn + fp)

    return f2_total / len(thresholds)
