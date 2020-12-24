import pandas as pd
from torch.utils.data import Dataset
import cv2
import numpy as np


class Melanoma3(Dataset):
    def __init__(self, data, more_pos=None, folds=None, tta=False, folder="train", transforms=None):

        self.prefix = "/".join(data.split("/")[:-1])
        self.data = pd.read_csv(data)
        self.transform = transforms
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
                        new_data = file
                        self.data = self.data.append(new_data)
                    elif new_folds.shape[0] == 30:
                        # 2018
                        new_data = file[file["tfrecord"].isin(new_folds[::2])]
                        self.data = self.data.append(new_data)

                        # 2019
                        new_data = file[file["tfrecord"].isin(new_folds[1::2])]
                        self.data = self.data.append(new_data)
                else:
                    self.data = self.data.append(file)

        self.length = self.data.shape[0]
        self.labels = (self.data["target"] > 0.5).tolist() if "target" in self.data.keys() else []

    def __len__(self):
        return self.length

    def visualize(self):
        folds = [self.data[self.data["fold"].isin([i]) & (self.data["source"] == "ISIC20")] for i in range(5)]
        for fold in folds:
            plt.hist(fold["age_approx"])
            plt.show()

        for fold in folds:
            plt.hist((fold["sex"] == "male").astype(np.float32))
            plt.show()

        for fold in folds:
            print((fold["target"] == 0).astype(np.float32).mean())
            plt.hist((fold["target"] == 0).astype(np.float32))
            plt.show()

        for fold in folds:
            data = fold.groupby("anatom_site_general_challenge").count()
            plt.bar(list(data.T.keys()), list(data["image_id"]))
            plt.show()

    def __getitem__(self, item):
        img = cv2.imread(os.path.join(self.prefix, self.folder, self.data["image_name"].iloc[item] + ".jpg"))
        if img is None:
            img = cv2.imread(os.path.join(self.prefix, "jpeg", self.data["image_name"].iloc[item] + ".jpg"))
        if img is None:
            img = cv2.imread(os.path.join(self.prefix, "train_2019", self.data["image_name"].iloc[item] + ".jpg"))
        if img is None:
            img = cv2.imread(os.path.join(self.prefix, "test", self.data["image_name"].iloc[item] + ".jpg"))
        target = np.array([float(self.data["target"].iloc[item])]) if "target" in self.data.keys() else \
        self.data["image_name"].iloc[item]
        if self.transform:
            if self.tta:
                imgs = []
                for i in range(self.tta):
                    imgs.append(self.transform(image=img.astype(np.uint8).copy())["image"].transpose(2, 0, 1))
                img = np.stack(imgs)
            else:
                img = self.transform(image=img.astype(np.uint8).copy())["image"].transpose(2, 0, 1)

        return {"input": img.astype(np.float32),
                "meta": self.data["image_name"].iloc[item],
                "y": target
                }
