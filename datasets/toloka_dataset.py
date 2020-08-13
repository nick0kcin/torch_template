import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import json
import cv2
from pycocotools.coco import COCO


def gaussian2d(shape, bias, sigma=(1, 1), scale=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m+1, -n:n+1]

    h = scale * np.exp(-((x + bias[1]) * (x + bias[1]) / sigma[1] ** 2 + (y + bias[0]) * (y + bias[0]) / sigma[0] ** 2))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h


def draw_gaussian(image, rect, shape_bias, sigma, scale=1):
    image[rect[0]:rect[2], rect[1]:rect[3]] = np.maximum(image[rect[0]:rect[2], rect[1]:rect[3]],
                                                         gaussian2d((rect[2] - rect[0], rect[3] - rect[1]),
                                                                    shape_bias, sigma, scale))
    return image


class TolokaDataset(Dataset):
    num_classes = 6
    label_dict = {"Boat": 0, "Bouy": 1, "Vessel": 2, "Millitary": 3, "Ice": 4, "Other": 5}
    super_class_dict = {"Boat": 0, "Bouy": 1, "Vessel": 0, "Millitary": 0, "Ice": 1, "Other": 1}
    MIN_OBJECT_AREA = 400
    mean = np.array([0.485, 0.456, 0.406], np.float32).reshape(1, 1, 3)
    std = np.array([0.229, 0.224, 0.225], np.float32).reshape(1, 1, 3)

    def __init__(self, path, path_mapping, transforms=None, augment=True, rotate=0, down_ratio=4, output_dim=512,
                 scales=(1, )):
        self.data = pd.read_csv(path, sep="\t")
        self.shapes = [(0, 0)] * self.data.shape[0]
        self.path_mappping = path_mapping
        self.down_ratio = down_ratio
        self.output_dim = output_dim
        self.augment = augment
        self.rotate = rotate
        self.Coco = None
        self.transforms = transforms
        self.scales = scales

    def summary(self):
        stats = np.zeros((self.num_classes,))
        for index, row in self.data.iterrows():
            try:
                anno = json.loads(row[3])
                for rect in anno:
                    try:
                        stats[self.label_dict.get(rect["annotation"], 5)] += 1
                    except KeyError:
                        pass
            except TypeError:
                pass
        return stats

    def coco(self, super_class=False):
        if not self.Coco:
            images = [{"file_name":  row[0], "id": i} for i, row in self.data.iterrows()]
            categories = [{"id": value, "name": key} for key, value in self.label_dict.items()]
            annotations = []
            anno_id = 0
            for i, row in self.data.iterrows():
                try:
                    anno = json.loads(row[3])
                except TypeError:
                    anno = []

                shape = cv2.imread(self.data.iloc[i, 0].replace(self.path_mappping[0], self.path_mappping[1])).shape
                for rect in anno:
                    bbox = list(self.relative_bbox2absolute(rect["data"], shape[:2]))
                    bbox = [int(bbox[0]), int(bbox[1]), int(bbox[2] - bbox[0]), int(bbox[3] - bbox[1])]
                    if bbox[2] * bbox[3] > self.MIN_OBJECT_AREA:
                        category_id = self.super_class_dict.get(rect["annotation"], 5) if super_class else\
                            self.label_dict.get(rect["annotation"], 5)
                        area = int(bbox[3] * bbox[2])
                        image_id = i
                        _id = anno_id
                        anno_id += 1
                        annotations.append({"bbox": bbox, "category_id": category_id,
                                            "area": area, "image_id": image_id, "id": _id, "iscrowd": 0, "score": 1.0})
            self.Coco = COCO()
            self.Coco.dataset = {"images": images, "categories": categories, "annotations": annotations}
            self.Coco.createIndex()
        return self.Coco

    @staticmethod
    def relative_bbox2absolute(rect, image_shape):
        box = (np.clip(rect["p1"]["y"], 0, 1) * image_shape[0], np.clip(rect["p1"]["x"], 0, 1) * image_shape[1],
               np.clip(rect["p2"]["y"], 0, 1) * image_shape[0], np.clip(rect["p2"]["x"], 0, 1) * image_shape[1])
        if box[0] > box[2]:
            box = box[2], box[1], box[0], box[3]
        if box[1] > box[3]:
            box = box[0], box[3], box[2], box[1]
        return box

    @staticmethod
    def _get_max_shape(anno):
        max_shape = [0, 0]
        for rect in anno:
            bbox = rect["data"]
            max_shape[0] = max(abs(bbox["p1"]["y"] - 0.5), abs(bbox["p2"]["y"] - 0.5), max_shape[0])
            max_shape[1] = max(abs(bbox["p1"]["x"] - 0.5), abs(bbox["p2"]["x"] - 0.5), max_shape[1])
        return max_shape

    def __len__(self):
        return self.data.shape[0] * len(self.scales)

    def __getitem__(self, ind):
        index = ind // len(self.scales)
        img_scale = self.scales[ind % len(self.scales)]
        img = cv2.imread(self.data.iloc[index, 0].replace(self.path_mappping[0], self.path_mappping[1]))
        if img_scale != 1:
            img = cv2.resize(img, dsize=(int(img.shape[1] / img_scale), int(img.shape[0] / img_scale)))
        if self.shapes[index] == (0, 0):
            self.shapes[index] = img.shape
        try:
            annotation = json.loads(self.data.iloc[index, 3])
        except TypeError:
            annotation = []
        num_objs = len(annotation)

        # for anno in annotation:
        #     box = self.relative_bbox2absolute(anno["data"], img.shape)
        #     cv2.rectangle(img, (int(box[1]), int(box[0])), (int(box[3]), int(box[2])), (255, 0, 0), 10)
        # cv2.imshow("123", img)
        # cv2.waitKey()

        if self.augment:
            image_w = self.output_dim
            image_h = self.output_dim

            angle = self.rotate * np.random.uniform(-1, 1)
            rotate_mat = cv2.getRotationMatrix2D((img.shape[1] / 2, img.shape[0] / 2), angle, 1)
            rect = cv2.transform(
                np.array([[[0, 0]], [[img.shape[1], 0]], [[0, img.shape[0]]], [[img.shape[1], img.shape[0]]]],
                         dtype=np.float32), rotate_mat)[:, 0, :]
            shift = np.min(rect, axis=0)
            rotate_mat[:, 2] -= shift

            choice = np.random.randint(0, num_objs + 1)
            border = int(self.output_dim / (2 * np.cos(np.deg2rad(angle))) + 0.5)

            if choice < num_objs:
                ann = annotation[choice]
                bbox = self.relative_bbox2absolute(ann['data'], img.shape)
                sigma = (bbox[2] - bbox[0], bbox[3] - bbox[1])
                area = sigma[0] * sigma[1]
                max_scale = min(min(img.shape[0:2]) / (2 * border), np.sqrt(area / self.MIN_OBJECT_AREA))
                scale = np.random.uniform(0.5, max(1 + 1e-06, max_scale))
                border = min(min(img.shape[0:2]) // 2, int(border * scale))
                c = ((bbox[2] + bbox[0]) / 2, (bbox[3] + bbox[1]) / 2)
                center_point = np.random.normal(c, sigma, (2,))
                center_point[0] = np.clip(center_point[0], border, img.shape[0] - border - 1)
                center_point[1] = np.clip(center_point[1], border, img.shape[1] - border - 1)
                center_point = cv2.transform(center_point, rotate_mat.reshape(1, 2, 3)).astype(np.int32)
                center_point = center_point[:, 0]
                s_low = max(abs(c[0] - center_point[0]), abs(c[1] - center_point[1]), int(border))
                s_high = min(img.shape[0] - center_point[0] - 1,
                             img.shape[1] - center_point[1] - 1, center_point[0], center_point[1])
                size = s_high if s_low >= s_high else np.random.randint(s_low,  int(s_high))
            else:
                c = (img.shape[0] / 2, img.shape[1] / 2)
                sigma = (max((img.shape[0] - 2 * border), 0) / 6, max(0, (img.shape[1] - 2 * border)) / 6)
                center_point = np.random.normal(c, sigma, (2,))
                center_point[0] = np.clip(center_point[0], border, img.shape[0] - border - 1)
                center_point[1] = np.clip(center_point[1], border, img.shape[1] - border - 1)
                center_point = cv2.transform(center_point, rotate_mat.reshape(1, 2, 3)).astype(np.int32)
                center_point = center_point[:, 0]
                s_low = self.output_dim // 2
                s_high = min(img.shape[0] - center_point[0] - 1,
                             img.shape[1] - center_point[1] - 1, center_point[0], center_point[1])
                size = s_high if s_low >= s_high else np.random.randint(s_low,  int(s_high))

            rect = cv2.transform(
                np.array([[[0, 0]], [[img.shape[1], 0]], [[0, img.shape[0]]], [[img.shape[1], img.shape[0]]]],
                         dtype=np.float32), rotate_mat)[:, 0, :]
            img = cv2.warpAffine(img, rotate_mat, (np.max(rect[:, 0]), np.max(rect[:, 1])), flags=cv2.INTER_LINEAR)
            sample = img[center_point[0] - size:center_point[0] + size,
                         center_point[1] - size: center_point[1] + size, :].copy()
            image = cv2.resize(sample, (self.output_dim, self.output_dim))
            trans_output = cv2.getAffineTransform(np.array([[center_point[0] - size, center_point[1] - size],
                                                            [center_point[0] + size, center_point[1] - size],
                                                            [center_point[0] - size, center_point[1] + size]],
                                                           dtype=np.float32),
                                                  np.array([[0, 0], [self.output_dim, 0], [0, self.output_dim]],
                                                           dtype=np.float32))
        else:
            trans_output = cv2.getRotationMatrix2D((0, 0), 0, 1)
            image_w = img.shape[0]
            image_h = img.shape[1]
            image = img

        # for anno in annotation:
        # # if idx != -1:
        #     bbox = self.relative_bbox2absolute(anno["data"], img.shape)
        #     center_point_g = np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32)
        #     center_point_g = cv2.transform(center_point_g.reshape(1,1,2), trans_output)
        #     c = np.sqrt(trans_output[0,0] ** 2 + trans_output[0,1] ** 2 )
        #     w, h = (bbox[3] - bbox[1]) / 2 * c, (bbox[2] - bbox[0]) / 2 * c
        #     bbox = [center_point_g[0,0,0] - h, center_point_g[0,0,1] -w, center_point_g[0,0,0] + h,
        #             center_point_g[0,0,1] + w]
        #     #bbox = [np.min(box[:, 0, 0]), np.min(box[:, 0, 1]), np.max(box[:, 0, 0]), np.max(box[:, 0, 1])]
        #     cv2.rectangle(image, (int(bbox[1]), int(bbox[0])), (int(bbox[3]), int(bbox[2])), (255, 0, 0), 10)
        #     cv2.circle(image, (center_point_g[0,0,1], center_point_g[0,0,0]), 40, (0, 255, 0), 10)
        # cv2.imshow("123", image)
        # cv2.waitKey()

        center_map = np.zeros((self.num_classes, image_w // self.down_ratio, image_h // self.down_ratio))
        wh_map = np.zeros((2, image_w // self.down_ratio, image_h // self.down_ratio))
        for anno in annotation:
            bbox = self.relative_bbox2absolute(anno["data"], img.shape)
            center_point_g = np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32)
            center_point_g = cv2.transform(center_point_g.reshape(1, 1, 2), trans_output)
            c = np.sqrt(trans_output[0, 0] ** 2 + trans_output[0, 1] ** 2)
            w, h = (bbox[3] - bbox[1]) / 2 * c, (bbox[2] - bbox[0]) / 2 * c
            unbounded_bbox = [center_point_g[0, 0, 0] - h, center_point_g[0, 0, 1] - w, center_point_g[0, 0, 0] + h,
                              center_point_g[0, 0, 1] + w]
            bbox = [0] * 4
            bbox[0::2] = np.clip(unbounded_bbox[0::2], 0, image_w)
            bbox[1::2] = np.clip(unbounded_bbox[1::2], 0, image_h)
            # gaussian_bias = (((bbox[0] + bbox[2]) / 2 - center_point_g[0, 0, 0]) / self.down_ratio,
            #                  ((bbox[1] + bbox[3]) / 2 - center_point_g[0, 0, 1]) / self.down_ratio)
            gaussian_bias = (0, 0)
            sigma = ((unbounded_bbox[2] - unbounded_bbox[0]) / (self.down_ratio * 3),
                     (unbounded_bbox[3] - unbounded_bbox[1]) / (self.down_ratio * 3))
            if (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) >= self.MIN_OBJECT_AREA\
                    and (bbox[2] - bbox[0]) > 4 and (bbox[3] - bbox[1]) > 4:
                bbox = [int(b / self.down_ratio) for b in bbox]
                class_index = self.label_dict.get(anno["annotation"], 5)
                center_map[class_index, :, :] = draw_gaussian(center_map[class_index, :, :], bbox, gaussian_bias, sigma)
                wh_map[0, :, :] = draw_gaussian(wh_map[0, :, :], bbox, gaussian_bias, sigma, (bbox[3] - bbox[1]) // 2)
                wh_map[1, :, :] = draw_gaussian(wh_map[1, :, :], bbox, gaussian_bias, sigma, (bbox[2] - bbox[0]) // 2)

        # map = np.zeros((wh_map.shape[1], wh_map.shape[2], 3))
        # map[:, :, 0] = wh_map[0, :, :]
        # map[:, :, 1] = wh_map[1, :, :]
        # map = cv2.resize((map).astype(np.uint8), (image_h, image_w))
        # cv2.imshow("321", map)
        # cv2.imshow("123", image)
        # cv2.waitKey()
        ship_center = center_map[0, :, :] + center_map[2, :, :] + center_map[3, :, :]
        obj_center = center_map[1, :, :] + center_map[4, :, :] + center_map[5, :, :]
        super_class_map = np.stack((ship_center, obj_center))
        if self.transforms:
            image = self.transforms(image)
        else:
            image = image.astype(np.float32).transpose(2, 0, 1) / 255

        # cv2.imshow("qwe", image.numpy().transpose(1,2,0))
        # cv2.waitKey()
        return {"input": image,
                "center": center_map.astype(np.float32),
                "dim": wh_map.astype(np.float32), "weight": self.data.iloc[index, -1].astype(np.float32),
                "super": super_class_map.astype(np.float32),
                "meta": {"name": self.data.iloc[index, 0].replace(self.path_mappping[0], self.path_mappping[1])}}
