import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import os
import cv2
import itertools
from collections import Counter
from albumentations import *
from image_utils import get_masks_from_polygon, draw_gaussian, masks_to_rot_bboxes, zipped_masks_to_rot_bboxes
from collections import namedtuple
import time

from matplotlib import pyplot as plt


class Melanoma(Dataset):

    def __init__(self, data, dublicates=None, folds=None, transform=None, pairs=False):
        cv2.startWindowThread()
        # cv2.namedWindow("qwe")
        self.prefix = "/".join(data.split("/")[:-1])
        self.pairs = pairs
        self.data = pd.read_csv(data)
        self.transform = transform
        self.folder = "512x512-dataset-melanoma/512x512-dataset-melanoma" if dublicates else "../512x512-test/512x512-test"
        if dublicates:
            self.dublicates = pd.read_csv(dublicates)
            self.dublictes2 = pd.read_csv("/media/nick/DATA/duplicates_13062020.csv")
        # d0 = self.data[self.data["image_id"].isin(self.dublicates["image_id0"])]
        # d1 = self.data[self.data["image_id"].isin(self.dublicates["image_id1"])]
        if dublicates is not None:
            self.data = self.data[self.data["fold"].isin(folds) & ~self.data["image_id"].isin(self.dublicates["image_id1"])].copy()
            images = []
            for i, row in self.dublictes2.iterrows():
                images.extend(row[0].split("."))
            self.data = self.data[~self.data["image_id"].isin(images)].copy()
            if len(folds) == 1:
                self.data = self.data[self.data["source"] == "ISIC20"].copy()

            # d2 = pd.read_csv("/media/nick/DATA/2020_Challenge_duplicates.csv")
            # d0 = self.data[self.data["image_id"].isin(d2["ISIC_id"])]
            # d1 = self.data[self.data["image_id"].isin(d2["ISIC_id_paired"])]
            # self.data = self.data[
            #     self.data["fold"].isin(folds) & ~self.data["image_id"].isin(d2["ISIC_id"])]
            # d0 = self.data[self.data["image_id"].isin(self.dublicates["image_id0"])]
            # d1 = self.data[self.data["image_id"].isin(self.dublicates["image_id1"])]
        if pairs:
            self.pos = self.data[self.data["target"] > 0]
            self.neg = self.data[self.data["target"] < 1]
            self.length = self.pos.shape[0]
        else:
            if "target" in self.data.keys():
                pos = self.data["target"].sum()
                self.weights = [self.data.shape[0] / pos  if val["target"] > 0 else 1 for i, val in self.data.iterrows()]
            #     self.length = 4 * int(pos) if len(folds) > 1 else self.data.shape[0]
            # else:
            self.length = self.data.shape[0]
        self.labels = self.data["target"].tolist() if "target" in self.data.keys() else []

    def __len__(self):
        return 4 * self.length if self.pairs else self.length


    def visualize(self):
        folds = [ self.data[self.data["fold"].isin([i]) & (self.data["source"] == "ISIC20")] for i in range(5)]
        for fold in folds:
            plt.hist(fold["age_approx"])
            plt.show()

        for fold in folds:
            plt.hist((fold["sex"]=="male").astype(np.float32))
            plt.show()


        for fold in folds:
            print((fold["target"]==0).astype(np.float32).mean())
            plt.hist((fold["target"]==0).astype(np.float32))
            plt.show()

        for fold in folds:
            data = fold.groupby("anatom_site_general_challenge").count()
            plt.bar(list(data.T.keys()),  list(data["image_id"]))
            plt.show()



    def __getitem__(self, item):
        if self.pairs:
            img_true = cv2.imread(os.path.join(self.prefix, self.folder, self.pos["image_id"].iloc[item // 4] + ".jpg"))
            false_idx = np.random.randint(0, self.neg.shape[0] - 1)
            img_false = cv2.imread(os.path.join(self.prefix, self.folder, self.neg["image_id"].iloc[false_idx] + ".jpg"))
            if self.transform:
                img_true = self.transform(image=img_true)
                img_false = self.transform(image=img_false)
            return {"input": np.stack((img_true["image"].transpose(2, 0, 1).astype(np.float32),
                                       img_false["image"].transpose(2, 0, 1).astype(np.float32))),
                    "y": np.array([[1.], [0.]])}
        else:
            img = cv2.imread(os.path.join(self.prefix, self.folder, self.data["image_id"].iloc[item] + ".jpg"))
            # if "target" not in self.data.keys():
            #     img = img[:, :, ::-1]
            target = np.array([float(self.data["target"].iloc[item])]) if "target" in self.data.keys() else self.data["image_id"].iloc[item]
            # plt.imshow(img)
            # plt.show()
            if self.transform:
                img = self.transform(image=img.astype(np.uint8).copy())

            # im = img["image"].copy()
            # plt.imshow(im)
            # plt.show()
            return {"input": img["image"].transpose(2, 0, 1).astype(np.float32),
                    "y": target
                         + np.random.random() * 0.05 * 2 * (0.5 - target.sum()) if not isinstance(target, str) else target
             }



class Melanoma2(Dataset):

    def __init__(self, data, folds=None, tta=False, folder="train", transform=None):
        cv2.startWindowThread()
        # cv2.namedWindow("qwe")
        self.prefix = "/".join(data.split("/")[:-1])
        self.data = pd.read_csv(data)
        self.transform = transform
        self.tta = tta
        self.folder = folder
        if "tfrecord" in self.data.keys():
            self.data = self.data[self.data["tfrecord"].isin(folds)].copy()
        self.length = self.data.shape[0]
        self.labels = self.data["target"].tolist() if "target" in self.data.keys() else []

    def __len__(self):
        return  self.length


    def visualize(self):
        folds = [ self.data[self.data["fold"].isin([i]) & (self.data["source"] == "ISIC20")] for i in range(5)]
        for fold in folds:
            plt.hist(fold["age_approx"])
            plt.show()

        for fold in folds:
            plt.hist((fold["sex"]=="male").astype(np.float32))
            plt.show()


        for fold in folds:
            print((fold["target"]==0).astype(np.float32).mean())
            plt.hist((fold["target"]==0).astype(np.float32))
            plt.show()

        for fold in folds:
            data = fold.groupby("anatom_site_general_challenge").count()
            plt.bar(list(data.T.keys()),  list(data["image_id"]))
            plt.show()



    def __getitem__(self, item):
        img = cv2.imread(os.path.join(self.prefix, self.folder, self.data["image_name"].iloc[item] + ".jpg"))
        # if "target" not in self.data.keys():
        #     img = img[:, :, ::-1]
        target = np.array([float(self.data["target"].iloc[item])]) if "target" in self.data.keys() else self.data["image_name"].iloc[item]
        # plt.imshow(img)
        # plt.show()
        if self.transform:
            if self.tta:
                imgs = []
                for i in range(self.tta):
                    imgs.append(self.transform(image=img.astype(np.uint8).copy())["image"].transpose(2, 0, 1))
                img = np.stack(imgs)
            else:
                img = self.transform(image=img.astype(np.uint8).copy())["image"].transpose(2, 0, 1)

        # im = img.transpose(1,2,0)
        # plt.imshow(im)
        # plt.show()
        return {"input": img.astype(np.float32),
                "meta": self.data["image_name"].iloc[item],
                "y": target
                     # + np.random.random() * 0.05 * 2 * (0.5 - target.sum()) if not isinstance(target, str) else target
         }


class Melanoma3(Dataset):

    def __init__(self, data, more_pos = None, folds=None, tta=False, folder="train", transform=None):
        cv2.startWindowThread()
        # cv2.namedWindow("qwe")
        self.prefix = "/".join(data.split("/")[:-1])
        self.data = pd.read_csv(data)
        self.transform = transform
        self.tta = tta
        self.folder = folder
        if "tfrecord" in self.data.keys():
            self.data = self.data[self.data["tfrecord"].isin(folds)].copy()
        if more_pos:
            for el in more_pos:
                file = pd.read_csv(el)
                if "tfrecord" in file.keys():
                    new_folds = np.array(sorted(file["tfrecord"].unique()))
                    new_folds = new_folds[new_folds > 0]
                    if new_folds.shape[0] == 15:
                        new_data = file #[file["tfrecord"].isin(new_folds[folds])]
                        self.data = self.data.append(new_data)
                    elif new_folds.shape[0] == 30:
                        # 2018
                        # new_data = file[file["tfrecord"].isin(new_folds[::2])]
                        # self.data = self.data.append(new_data)

                        #2019
                        new_data = file[file["tfrecord"].isin(new_folds[1::2])]
                        self.data = self.data.append(new_data)
                else:
                    self.data = self.data.append(file)

        self.length = self.data.shape[0]
        self.labels = (self.data["target"] > 0.5).tolist() if "target" in self.data.keys() else []

    def __len__(self):
        return  self.length


    def visualize(self):
        folds = [ self.data[self.data["fold"].isin([i]) & (self.data["source"] == "ISIC20")] for i in range(5)]
        for fold in folds:
            plt.hist(fold["age_approx"])
            plt.show()

        for fold in folds:
            plt.hist((fold["sex"]=="male").astype(np.float32))
            plt.show()


        for fold in folds:
            print((fold["target"]==0).astype(np.float32).mean())
            plt.hist((fold["target"]==0).astype(np.float32))
            plt.show()

        for fold in folds:
            data = fold.groupby("anatom_site_general_challenge").count()
            plt.bar(list(data.T.keys()),  list(data["image_id"]))
            plt.show()



    def __getitem__(self, item):
        img = cv2.imread(os.path.join(self.prefix, self.folder, self.data["image_name"].iloc[item] + ".jpg"))
        if img is None:
            img = cv2.imread(os.path.join(self.prefix, "jpeg", self.data["image_name"].iloc[item] + ".jpg"))
        if img is None:
            img = cv2.imread(os.path.join(self.prefix, "train_2019", self.data["image_name"].iloc[item] + ".jpg"))
        if img is None:
            img = cv2.imread(os.path.join(self.prefix, "test", self.data["image_name"].iloc[item] + ".jpg"))
        # if "target" not in self.data.keys():
        #     img = img[:, :, ::-1]
        target = np.array([float(self.data["target"].iloc[item])]) if "target" in self.data.keys() else self.data["image_name"].iloc[item]
        # plt.imshow(img)
        # plt.show()
        if self.transform:
            if self.tta:
                imgs = []
                for i in range(self.tta):
                    imgs.append(self.transform(image=img.astype(np.uint8).copy())["image"].transpose(2, 0, 1))
                img = np.stack(imgs)
            else:
                img = self.transform(image=img.astype(np.uint8).copy())["image"].transpose(2, 0, 1)

        # im = img.transpose(1,2,0)
        # plt.imshow(im)
        # plt.show()
        return {"input": img.astype(np.float32),
                "meta": self.data["image_name"].iloc[item],
                "y": target
                     # + np.random.random() * 0.05 * 2 * (0.5 - target.sum()) if not isinstance(target, str) else target
         }