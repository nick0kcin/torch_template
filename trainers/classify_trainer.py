import os
import shutil

import cv2
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances
from tqdm import tqdm

from image_utils import ssim
from .base_trainer import BaseTrainer


class ClassifyTrainer(BaseTrainer):
    def __init__(self, model, losses, loss_weights, metrics, teacher=None, optimizer=None, num_iter=-1, print_iter=-1,
                 device=None,
                 batches_per_update=1):
        super(ClassifyTrainer, self).__init__(model, losses, loss_weights, metrics, teacher, optimizer, num_iter,
                                              print_iter,
                                              device, batches_per_update)

    def predict(self, dataloader, tta=None):
        return super(ClassifyTrainer, self)._predict(dataloader,
                                                     tta, label_mapper=lambda x: x["meta"])

    def features(self, dataloader):
        return super(ClassifyTrainer, self)._predict(dataloader, label_mapper=lambda x: x["meta"])

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
                                                                          label_mapper=lambda x: (x["y"], x["meta"])),
                                    total=len(dataloader)):
            targets.extend([el.view(-1).cpu().numpy().tolist() for el in pred[0].values()][0])
            names.extend(name[0].view(-1).cpu().numpy().tolist())
            n.extend(name[1])
        res["image_name"] = n
        res["target"] = targets
        res.to_csv(file, index=None)
        return targets, names

    def pca(self, dataloader_fit, dataloader_predict, components):
        ipca = PCA(n_components=components)  # , batch_size=len(dataloader_fit))
        first = None
        for img, pred, name in self.features(dataloader_fit):
            first = pred[0]["y"].cpu().numpy() if first is None else np.row_stack((first, pred[0]["y"].cpu().numpy()))

        ipca.fit(first)
        names = []
        features = None
        for img, pred, name in self.predict(dataloader_predict):
            names.extend(name)
            features = pred[0]["y"].cpu().numpy() if features is None else np.row_stack(
                (features, pred[0]["y"].cpu().numpy())
            )

        return names, features

    def find_optimal_clusters(self, dataloader):
        names, features = self.pca(dataloader, dataloader, 512)
        distes = []
        for i in range(1, 102, 10):
            algo = KMeans(i, init="k-means++")
            dist = algo.fit_transform(features)
            dist = dist.min(axis=1).mean()
            distes.append(dist)
        plt.plot(distes)
        plt.show()

    def cluster(self, dataloader, clusters):
        names, features = self.pca(dataloader, dataloader, 1024)
        algo = KMeans(clusters, algorithm="elkan", init="random", random_state=42)
        dist = algo.fit_transform(features)
        dist = dist.argmin(axis=1)
        for i in range(clusters):
            os.mkdir(f"{'/'.join(dataloader.dataset.path.split('/')[:-1])}/{i}")
        for idx, i in enumerate(dist):
            shutil.copy2(names[idx],
                         f"{'/'.join(dataloader.dataset.path.split('/')[:-1])}/{i}/{names[idx].split('/')[-1]}")

    def top_n_pairs(self, dataloader, n=100):
        names, features = self.pca(dataloader, dataloader, 1024)
        dist = pairwise_distances(features, metric="cosine")
        mn = dist.flatten()
        mn.sort()
        top_dist = mn[dist.shape[0]:dist.shape[0] + 2 * n]
        for i in range(n):
            pos = (dist == top_dist[2 * i]).nonzero()
            shutil.copy2(names[pos[0][0]],
                         f"{'/'.join(dataloader.dataset.path.split('/')[:-1])}/PAIRS{i}_{names[pos[0][0]].split('/')[-1]}")
            shutil.copy2(names[pos[0][1]],
                         f"{'/'.join(dataloader.dataset.path.split('/')[:-1])}/PAIRS{i}_{names[pos[0][1]].split('/')[-1]}")

    def dbscan(self, dataloader):
        names, features = self.pca(dataloader, dataloader, 1024)
        results = []
        for eps in [0.4]:
            algo = DBSCAN(eps=eps, min_samples=2, metric="cosine")
            dist = algo.fit_predict(features)
            for i in range(int(dist.max()) + 1):
                if (dist == i).sum() > 12:
                    dist[dist == i] = -1
            results.append(dist)

        results = np.stack(results)
        clusters = [tuple() for i in range(results.shape[1])]
        for i in range(results.shape[1]):
            pos = (results[:, i] >= 0).nonzero()
            if len(pos[0]):
                clusters[i] = pos[0][0], results[pos[0], i][0]
        for i in np.unique(clusters):
            if i:
                os.mkdir(f"{'/'.join(dataloader.dataset.path.split('/')[:-1])}/{i[0]}_{i[1]}")
        for idx, i in enumerate(clusters):
            if len(i) == 2:
                shutil.copy2(names[idx],
                             f"{'/'.join(dataloader.dataset.path.split('/')[:-1])}/{i[0]}_{i[1]}/{names[idx].split('/')[-1]}")

    def test(self, dataloader, verbose=0, tta=None):
        return super()._test(dataloader, "y", tta,
                             verbose=verbose
                             )

    def check_quality(self, dataloader):
        names = []
        ssims_median = []
        ssims_gauss = []
        for data in dataloader.dataset:
            image = cv2.cvtColor(data["input"].transpose(1, 2, 0).astype(np.uint8), cv2.COLOR_BGR2GRAY)
            median = cv2.medianBlur(image, 3)
            gauss = cv2.GaussianBlur(image, (7, 7), 1.5)
            ssims_gauss.append(ssim(image, gauss))
            ssims_median.append(ssim(image, median))
            names.append(data["meta"])

        file = pd.DataFrame(columns=["image_name", "gauss_score", "median_score"])
        file["image_name"] = names
        file["gauss_score"] = ssims_gauss
        file["median_score"] = ssims_median
        file.to_csv("/media/nick/57E900E38EEA892E11/dataset_flatten/ssim.csv", index=False)
