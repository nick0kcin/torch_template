from .base_trainer import BaseTrainer
import pandas as pd
from sklearn.decomposition import IncrementalPCA
from sklearn.cluster import KMeans
from tqdm import tqdm

from albumentations import Flip, ShiftScaleRotate, Normalize
import cv2
import numpy as np
from torch.nn.functional import max_pool2d
import functools
from copy import copy
from torch import Tensor, atan2
from transforms.MedianBlur import MedianBlur
from torchvision import transforms

class ClassifyTrainer(BaseTrainer):
    def __init__(self, model, losses, loss_weights, metrics, teacher=None, optimizer=None, num_iter=-1, print_iter=-1, device=None,
                 batches_per_update=1):
        super(ClassifyTrainer, self).__init__(model, losses, loss_weights, metrics, teacher, optimizer, num_iter, print_iter,
                                              device, batches_per_update)

    def predict(self, dataloader, tta=None):
        return super(ClassifyTrainer, self)._predict(dataloader,
                                                      tta, label_mapper=lambda x: x["meta"])


    def features(self, dataloader):
        return super(ClassifyTrainer, self)._predict(dataloader)


    def predict_file(self, dataloader, file_name):
        res = pd.DataFrame(columns=["image_name", "target"])
        targets = []
        names = []
        for img, pred, name in tqdm(self.predict(dataloader), total=len(dataloader)):
            targets.extend([el.view(-1).cpu().numpy().tolist() for el in pred[0].values()][0])
            names.extend(name)
        res["image_name"] = names
        res["target"] = targets
        res.to_csv(file_name, index=None)
        return res

    def predict_partial(self, dataloader, file):
        res = pd.DataFrame(columns=["image_name", "target"])
        targets = []
        names = []
        n = []
        for img, pred, name in tqdm(super(ClassifyTrainer, self)._predict(dataloader,
                                                                          label_mapper=lambda x: (x["y"], x["meta"])), total=len(dataloader)):
            targets.extend([el.view(-1).cpu().numpy().tolist() for el in pred[0].values()][0])
            names.extend(name[0].view(-1).cpu().numpy().tolist())
            n.extend(name[1])
        res["image_name"] = n
        res["target"] = targets
        res.to_csv(file, index=None)
        return  targets, names


    def pca(self, dataloader_fit, dataloader_predict, components):
        ipca = IncrementalPCA(n_components=components, batch_size=dataloader_fit.batch_size)
        for img, pred, name in self.features(dataloader_fit):
            ipca.partial_fit(pred.cpu().numpy())


        names = []
        features = []
        for img, pred, name in self.predict(dataloader_predict):
            names.append(name)
            features.append(pred.cpu().numpy())

        return names, features


    def find_optimal_clusters(self, dataloader):
        names, features = self.pca(dataloader, dataloader, 32)
        distes = []
        for i in range(5, 40, 5):
            algo = KMeans(i)
            dist = algo.fit_transform(features)
            dist = (dist.min(axis=1) ** 2).sum()
            distes.append(dist)

    def test(self, dataloader, verbose=0, tta=None):
        return super()._test(dataloader, "y", tta,
                             verbose=verbose
                             )


