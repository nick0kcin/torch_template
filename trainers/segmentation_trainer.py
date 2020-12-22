import os
import cv2
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from .base_trainer import BaseTrainer


class SegmentationTrainer(BaseTrainer):
    def __init__(self, model, losses, loss_weights, metrics, teacher=None, optimizer=None, num_iter=-1, print_iter=-1,
                 device=None,
                 batches_per_update=1):
        super(SegmentationTrainer, self).__init__(model, losses, loss_weights, metrics, teacher, optimizer, num_iter,
                                                  print_iter,
                                                  device, batches_per_update)

    def predict(self, dataloader, tta=None):
        return super(SegmentationTrainer, self)._predict(dataloader,
                                                         tta, label_mapper=lambda x: x["meta"])

    def predict_masks(self, dataloader, destination, fold):
        names = []
        scores_std = []
        losses_mean = []
        losses_std = []
        self.model_with_loss = self.model_with_loss.eval()
        torch.cuda.empty_cache()
        with tqdm(dataloader, total=len(dataloader)) as bar_object:
            for item, el in enumerate(bar_object):
                # if len(el["meta"][0].split("mv")) == 1:
                el = self.to(el)
                inp = el["input"]
                # print(inp.shape)
                if len(inp.shape) > 4:
                    tta = inp.shape[1]
                    inp = inp.view(-1, inp.shape[-3], inp.shape[-2], inp.shape[-1])
                with torch.no_grad():
                    if inp.shape[-1] * inp.shape[-2] <= 440 * 440:
                        res, loss_value, loss_stats = self.model_with_loss(el)
                    else:
                        try:
                            _res_mask = []
                            loss_values_ = []
                            for i in range(inp.shape[0]):
                                res_, loss_value_, loss_stats_ = self.model_with_loss(
                                    {"input": el["input"][:, i, :, :, :],
                                     "mask": el["mask"][:, i, :, :, :]})
                                _res_mask.append(res_["mask"])
                                loss_values_.append(loss_value_)
                            res = {"mask": torch.cat(_res_mask, dim=0)}
                            loss_value = torch.cat(loss_values_, dim=0)
                            # print(loss_value.shape, res["mask"].shape)
                        except RuntimeError:
                            print(el["input"].shape, i)

                mask = res["mask"][:, :, :el["mask"].shape[-2], :el["mask"].shape[-1]].sigmoid().mean(0)
                std = res["mask"][:, :, :el["mask"].shape[-2], :el["mask"].shape[-1]].sigmoid().std(0)
                std_cost = (std > 0.2).float().sum(-1).sum(-1) / (std.shape[-1] * std.shape[-2])
                loss_mean = loss_value.cpu().mean(0)
                loss_std = loss_value.cpu().std(0)

                img = mask.cpu().numpy().transpose(1, 2, 0)
                img_ = cv2.resize(img, (max(el["mask"].shape[-1], el["size"][1].item()),
                                        max(el["mask"].shape[-2], el["size"][0].item())))
                img = img_[:el["size"][0].item(), :el["size"][1].item()]
                std_ = std.cpu().numpy().transpose(1, 2, 0)
                std__ = cv2.resize(std_, (max(el["mask"].shape[-1], el["size"][1].item()),
                                          max(el["mask"].shape[-2], el["size"][0].item())))
                std_ = std__[:el["size"][0].item(), :el["size"][1].item()]

                np.save(os.path.join(destination, "img", el["meta"][0].split("/")[-1].replace(".png", ".npy")),
                        img)
                np.save(os.path.join(destination, "std", el["meta"][0].split("/")[-1].replace(".png", ".npy")),
                        std_)

                losses_mean.append(loss_mean.cpu().numpy())
                losses_std.append(loss_std.cpu().numpy())
                names.append(el["meta"][0])
                scores_std.append(std_cost.cpu().numpy())

        df = pd.DataFrame(columns=["name", "std"])
        df["name"] = names
        for i in range(losses_mean[0].shape[0]):
            df[f"mean{i}"] = [v[i] for v in losses_mean]
            df[f"std{i}"] = [v[i] for v in losses_std]
            df[f"scores{i}"] = [v[i] for v in scores_std]
        df.to_csv(os.path.join(destination, f"scores{fold}.csv"), index=False)

    def visualize(self, dataloader, models=None):
        if models:
            for i, model in enumerate(models):
                models[i] = model.eval().cuda()
        self.model_with_loss.eval()
        self.model_with_loss.module.model.eval()
        for el in dataloader:
            self.to(el)
            with torch.no_grad():
                if models:
                    res = []
                    for model in models:
                        rs = model(el[0])
                        res.append(rs[-1]["mask"][0, :, :, :].sigmoid().cpu().numpy().transpose(1, 2, 0))
                    return None, np.mean(res, axis=0)
                else:
                    res = self.model_with_loss.module.model(el[0])
                    return None, res[-1]["mask"][0, :, :, :].sigmoid().cpu().numpy().transpose(1, 2, 0)
