from .f2_instance_seg import f2
from utils import create_instance
from sklearn.metrics import roc_auc_score


def create_metric(name, **kwargs):
    return globals()[name]


class MetricsHandler:
    def __init__(self, metrics, verbose=0):
        self.metrics = metrics
        self.verbose = verbose

    def __call__(self, data_generator):
        result = {metric: 0 for metric in self.metrics}
        length = 0
        if result:
            for pred, gt in data_generator:
                for metric, func in self.metrics.items():
                    if isinstance(gt, list) and isinstance(pred, list) and len(gt) == len(pred):
                        for gt_mask, pred_mask in zip(gt, pred):
                            val = func(gt_mask, pred_mask)
                            result[metric] += val
                        length += len(gt)
                    else:
                        val = func(gt[0, :, :, :] if gt.nelement() else gt, pred[0])
                        result[metric] += val
                        length += 1
                if self.verbose:
                    print({metric: value / length for metric, value in result.items()})
            return {metric: value / length for metric, value in result.items()}
        return result
