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


class Dota(Dataset):
    def __init__(self, path, label_path, crop=True, transform=None):
        cv2.startWindowThread()
        self.path = path
        self.label_path = label_path
        self.class_mapping = {
            'small-vehicle': 'small-vehicle', 'tennis-court': 'bomb-goal', 'large-vehicle': 'large-vehicle',
            'plane': 'plane', 'storage-tank': 'storage-tank', 'ship': 'ship', 'harbor': 'harbor',
            'ground-track-field': "bomb-goal", 'soccer-ball-field': "bomb-goal", 'swimming-pool': "bomb-goal",
            'baseball-diamond': "bomb-goal", 'roundabout': 'bomb-goal',
            'basketball-court': "bomb-goal", 'bridge': "bridge", 'helicopter': 'helicopter',
            'container-crane': 'bomb-goal'
        }
        self.gen_labels(label_path)
        self.down_ratio = 4
        self.data_cols = {"pos": 2, "wh": 2, "bias": 2, "angle": 1}
        weights = {key: (max(self.classes_dict.values()) / val) ** 2 for key, val in self.classes_dict.items()}
        self.sample_weights = np.array([sum([val * weights[key] for key,val in counter.items()]) +1e-07
                               for counter in self.sample_labels])
        self.sample_weights /= sum(self.sample_weights)
        # self.exp = {cls: sum([ v * self.sample_labels[i][cls] for i, v in enumerate(self.sample_weights)]) for cls in self.classes_dict}
        self.class_idx = {cls: i for i, cls in enumerate(sorted(self.classes_dict.keys() - {"bomb-goal"}))}
        self.length = 2 * len(self.images)
        self.max_objects = max(self.num_objects)
        self.transforms = transform
        # self.crop_size = (512, 512)
        self.crop = crop

    def gen_labels(self, label_path, mapping=True):
        self.labels = {}
        self.classes_dict = Counter()
        self.images = []
        self.sample_labels = []
        self.num_objects = []
        for name in os.listdir(label_path):
            self.images.append(name)
            self.sample_labels.append(Counter())
            with open(os.path.join(label_path, name), "r") as file:
                lines = [line.replace(",", "").split() for line in file if ":" not in line]
                self.num_objects.append(len(lines))
                data = [(np.array(list(map(float, line[0:8]))).reshape((4, 2)),
                         self.class_mapping[line[8]] if mapping else line[8]) for line in lines]
                for _, class_name in data:
                    self.classes_dict.update({class_name})
                    self.sample_labels[-1].update({class_name})
                # data = [(get_rotated_rect(points), class_name) for points, class_name in data]
                self.labels.update({name: data})
        self.images.sort()

    def cut_big_image(self, img, objects, name, size=768, stride=0.7):
        last = None
        for i, (sx, sy) in enumerate(
                itertools.product(range(size, img.shape[0] + size - 1, int(size * stride)),
                                  range(size, img.shape[1] + size - 1, int(size * stride)))):
            x, y = min(sx, img.shape[1]), min(sy, img.shape[0])
            nx, ny = max(x - size, 0), max(y - size, 0)
            if last is None:
                last = np.array([x,y,nx,ny])
            elif np.abs(np.array([x,y,nx,ny]) - last).mean() < 20:
                continue
            crop = img[ny: y, nx: x, :].copy()
            vertex = np.array([nx, ny])
            crop_rect = np.array([[nx, ny],
                                  [nx, y],
                                  [x, y],
                                  [x, ny]], dtype=np.float32)
            good_objects = [(obj - vertex, class_label)
                            for obj, class_label in objects
                            if cv2.intersectConvexConvex(obj.astype(np.float32),
                                                         crop_rect)[0] > 0.5 * cv2.contourArea(obj.astype(np.float32))]
            assert crop.shape == (min(img.shape[0], size), min(img.shape[1], size), 3)
            cv2.imwrite(os.path.join(self.path, f"cutted_images/{name}_{i}.png"), crop)
            with open(os.path.join(self.path, f"cutted_labels/{name}_{i}.txt"), "w") as file:
                file.write("\n".join([f"{str(obj.reshape(1, 8)[0,:].tolist())[1:-1]}, {class_label}"
                                      for obj, class_label in good_objects]))
            # self.visualize(crop, good_objects)
            # self.visualize(crop, [(obj - vertex, class_label)
            #                 for obj, class_label in objects], name="eta")

    def __len__(self):
        return len(self.images)

    # def crop_image(self, img, objects):
    #     centroids = []
    #     if self.transforms:
    #         transformed = self.transforms(image=img, points=objects)
    #         if "points" in transformed and transformed["points"] is not None:
    #             for points in transformed["points"]:
    #                 centroids.append(points.mean(axis=0))
    #             objects = [obj for i, obj in enumerate(objects)
    #                        if centroids[i][0] < img.shape[1] and centroids[i][1] < img.shape[0]]
    #             centroids = [centroid for centroid in centroids
    #                          if centroid[0] < img.shape[1] and centroid[1] < img.shape[0]]
    #         img = transformed["image"]
    #     else:
    #         for obj in objects:
    #             centroids.append(obj.mean(axis=0))
    #     if not self.crop:
    #         return img, objects
    #     nd_crop = np.array(self.crop_size)
    #     nd_size = np.array(img.shape[1::-1])
    #     choise = None
    #     if np.random.rand() > 0.3 and objects:
    #         choise = np.random.randint(0, len(objects) - 1) if len(objects) > 1 else 0
    #         center = np.clip(np.random.normal(
    #         centroids[choise], nd_crop / 28), nd_crop / 2, nd_size - nd_crop / 2 - 1)
    #     else:
    #         center = np.clip(np.random.normal(nd_size / 2, nd_size / 4), nd_crop / 2, nd_size - nd_crop / 2 - 1)
    #     center = center.astype(np.uint32)
    #     img = img[center[1] - self.crop_size[1] // 2: center[1] + self.crop_size[1] // 2,
    #           center[0] - self.crop_size[0] // 2: center[0] + self.crop_size[0] // 2, :].copy()
    #     nobjects = [np.clip(obj, [center[0] - self.crop_size[0] // 2, center[1] - self.crop_size[1] // 2],
    #           [center[0] + self.crop_size[0] // 2, center[1] + self.crop_size[1] // 2]) for obj in objects]
    #     # assert (not choise) or cv2.contourArea(nobjects[choise].astype(np.float32)[:, None, :]) > 5,
    #     # (center, centroids[choise]) if choise else None
    #     nobjects = [obj for obj in nobjects if cv2.contourArea(np.float32(obj)[:, None, :]) > 5]
    #     return img, nobjects

    @staticmethod
    def visualize(img, objects, name="img"):
        # img = self.transforms(image=img)["image"]
        for obj, _ in objects:
            cv2.drawContours(img, np.int0(obj[None, :, :]), 0, (255, 0, 255), 3)
        cv2.imshow(name, img)
        # cv2.imwrite(os.path.join("/app", name), img)
        cv2.waitKey()

    @staticmethod
    def visualize_boxes(img, objects, name="img"):
        # img = self.transforms(image=img)["image"]
        for obj in objects:
            points = cv2.boxPoints(obj)
            cv2.drawContours(cv2.UMat(img), np.int0(points[None, :, :]), 0, (255, 0, 0))
        cv2.imshow(name, img)
        cv2.waitKey()

    def debug_big(self):
        for name in [1179, 1315, 1332, 1397, 1432, 1604, 2271, 2408, 2446, 2488, 2541, 2645]:
            full_name = os.path.join(self.path, "images", f"P{name}.png")
            img = cv2.imread(full_name)
            label_path = os.path.join(self.path, "labelTxt-v1.5", "DOTA-v1.5_val")
            with open(os.path.join(label_path, f"P{name}.txt"), "r") as file:
                lines = [line.replace(",", "").split() for line in file if ":" not in line]
                self.num_objects.append(len(lines))
                data = [(np.array(list(map(float, line[0:8]))).reshape((4, 2)),
                         self.class_mapping[line[8]]) for line in lines]
            self.visualize(img, data, f"P{name}.png")

    def get_masks(self, objects, img_size):
        center_mask = np.zeros((len(self.class_idx), img_size[0] // self.down_ratio, img_size[1] // self.down_ratio))
        data_type = namedtuple("dims", self.data_cols.keys())
        dims = {value: [] for value in self.data_cols}
        for rect, cls in objects:
            bbox = cv2.boundingRect(cv2.boxPoints(rect))
            down_rect = ((int(rect[0][0] // self.down_ratio + 0.5), int(rect[0][1] // self.down_ratio + 0.5)),
                         (max(3, rect[1][0] // self.down_ratio), max(3, rect[1][1] // self.down_ratio)),
                         rect[2])
            df = (rect[0][0] - int(down_rect[0][0]) * self.down_ratio,
                  rect[0][1] - int(down_rect[0][1]) * self.down_ratio)

            down_bbox = [down_rect[0][0] - bbox[2] // 2, down_rect[0][1] - bbox[3] // 2,
                         2 * (bbox[2] // 2) + 1, 2 * (bbox[3] // 2) + 1]
            cls_idx = self.class_idx[cls]
            center_mask[cls_idx, :, :] = draw_gaussian(center_mask[cls_idx, :, :], down_rect, down_bbox)

            assert center_mask[cls_idx, int(down_rect[0][1]), int(down_rect[0][0])] == 1
            assert int(down_rect[0][0]) * self.down_ratio + df[0] == rect[0][0]
            assert int(down_rect[0][1]) * self.down_ratio + df[1] == rect[0][1]

            item = data_type(down_rect[0], rect[1], df, ((90 + rect[2]) if rect[2] <= 0 else 90,))
            try:
                assert 0 <= item.angle[0] <= 90
            except AssertionError:
                print(rect[2])
            for key, value in item._asdict().items():
                assert isinstance(value, tuple) or isinstance(value, list)
                dims[key].append(value)

        return center_mask, dims

    def __getitem__(self, item):
        full_name = os.path.join(self.path, "cutted_images", self.images[item].replace(".txt", ".png"))
        # print(full_name)
        img = cv2.imread(full_name)
        if img is None:
            os.remove(os.path.join(self.path, "cutted_labels", self.images[item]))
            return item
            # raise ValueError(f"{full_name} not found")
        objects = self.labels[self.images[item]]
        objects_lab = [lab for _, lab in objects]
        # self.cut_big_image(img, objects, self.images[item][:-4])

        try:
            # start = time.clock()
            masks = np.load(os.path.join(self.path, "masks", self.images[item].replace(".txt", ".npy")))
            # t = time.clock() - start
            # print("load", t)
        except:
            # start = time.clock()
            masks = np.stack(get_masks_from_polygon(objects, img.shape[:2]), axis=-1)
            # t = time.clock() - start
            # if t > 0.025:
            #     np.save(os.path.join(self.path, "masks", self.images[item].replace(".txt", ".npy")), masks)
            # print("make", t)
        # # #
        # # if len(objects) > 30:
        # #     np.save(os.path.join(self.path, "masks", self.images[item].replace(".txt", ".npy")), masks)
        # #
        transformed = self.transforms(image=img,
                                      mask=masks)
        bboxes = zipped_masks_to_rot_bboxes(transformed["mask"], len(objects_lab))
        img = transformed["image"]
        assert len(bboxes) == len(objects_lab)
        bboxes = [(box, label) for box, label in zip(bboxes, objects_lab) if box and label != "bomb-goal"]
        center_mask, dimension_data = self.get_masks(bboxes, img.shape[:2])

        # img = np.zeros(shape=(512, 512, 3))
        # center_mask = np.zeros(shape=(len(self.class_idx),) + tuple([i //self.down_ratio for i in img.shape[:2]]))
        # data_type = namedtuple("dims", self.data_cols.keys())
        # dimension_data = {value: [] for value in self.data_cols}
        # for i in range(1):
        #     item = data_type((np.random.randint(0, 60), np.random.randint(0, 60)), (10, 10), (1, 1), (0,))
        #     for key, value in item._asdict().items():
        #         assert isinstance(value, tuple) or isinstance(value, list)
        #         dimension_data[key].append(value)

        ret = {"input": img.transpose(2, 0, 1).astype(np.float32),
               "center": center_mask.astype(np.float32),
               "meta": full_name
               }
        for key, val in dimension_data.items():
            dimension_data[key] = np.row_stack(
                (np.array(val, dtype=np.int64 if key == "pos" else np.float32),
                 -np.ones((self.max_objects - len(val),
                           self.data_cols[key]), dtype=np.int64 if key == "pos" else np.float32)
                 )) if val else -np.ones((self.max_objects,
                                          self.data_cols[key]), dtype=np.int64 if key == "pos" else np.float32)
        ret.update(dimension_data)

        # if not (item % 2):
        #     self.cut_big_image(img, objects, self.images[item][:-4])
        # self.visualize(img, objects)
        # self.visualize_boxes(img, bboxes, "boxes")
        return ret
        # return item
