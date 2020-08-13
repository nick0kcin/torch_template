from .f2_instance_seg import f2
from utils import create_instance
from sklearn.metrics import roc_auc_score


def create_metric(name, **kwargs):
    return globals()[name]#create_instance(name, globals(), **kwargs)


class MetricsHandler:
    def __init__(self, metrics, verbose=0):
        self.metrics = metrics
        self.verbose = verbose

    def __call__(self, data_generator):
        result = []
        gt = []
        for pred in data_generator:
            # return pred[1][0]
            result.extend(pred[1][0]["y"].sigmoid().cpu().numpy().tolist())
            # result.extend([el.sigmoid().item() for el, in pred[1][0].items()])
            # gt.extend(([el["y"].item() for el in pred[2].items()]))
            #
            gt.extend((pred[2] > 0.5).float().cpu().numpy().flatten().tolist())
            # gt.extend(pred[2])

        val = roc_auc_score(gt, result)
        return float(val)
        return result, gt
    # def __call__(self, data_generator):
    #     result = {metric: 0 for metric in self.metrics}
    #     length = 0
    #     if result:
    #         for pred in data_generator:
    #             for metric, func in self.metrics.items():
    #                 if isinstance(gt, list) and isinstance(pred, list) and len(gt) == len(pred):
    #                     for gt_mask, pred_mask in zip(gt, pred):
    #                         val = func(gt_mask, pred_mask)
    #                         result[metric] += val
    #                     length += len(gt)
    #                 else:
    #                     val = func(gt[0, :, :, :] if gt.nelement() else gt, pred[0])
    #                     result[metric] += val
    #                     length += 1
    #                 # if gt.sum() > 0 and (gt.shape[1] == 1 and pred.shape[0] == 1):
    #                 #     print("inside", ((gt * pred) > 0).sum().float() / ((gt + pred) > 0).sum().float() + 1e-09)
    #             if self.verbose:
    #                 print({metric: value / length for metric, value in result.items()})
    #         return {metric: value / length for metric, value in result.items()}
    #     return result
