import itertools
from collections import namedtuple

import cv2
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

from image_utils import rle_decode, draw_gaussian


# np.seterr(divide='raise', invalid='raise')


class AirBusDataset(Dataset):

    def __init__(self, path, augment=False, crop_size=256, transforms=None, segmentation=False, angle_encoding="sin"):
        self.data = pd.read_csv(path)
        self.images = list(set(self.data.iloc[:, 0]))
        self.sizes = self.data.dropna()['ImageId'].dropna().value_counts()
        self.sizes_all = self.data['ImageId'].dropna().value_counts()
        self.max_objects = max(self.sizes)
        neg_weight = self.sizes.sum() * 0.2 / (len(self.images) - self.sizes.shape[0])
        self.length = int(1.2 * self.sizes.shape[0]) if augment else len(self.images)
        dict_sizes = dict(self.sizes)
        self.weights = np.array([dict_sizes.get(file, neg_weight) for file in self.images])
        self.weights /= self.weights.sum()
        self.img_size = (768, 768)
        self.crop_size = (crop_size, crop_size)
        self.augment = augment
        self.stat = [0, 0]
        self.stat_s = []
        self.down_ratio = 2
        self.transforms = transforms
        self.data_cols = {"pos": 2, "wh": 2, "bias": 2, "angle": 2 if angle_encoding == "sin" else 1}
        self.segmentation = segmentation
        self.angle_encoding = angle_encoding

    @property
    def size(self):
        return self.crop_size if self.augment else self.img_size

    def abj_rect(self, decoded, original):
        contours = decoded.nonzero()
        contours = np.array(list(zip(contours[1], contours[0])))[:, None, :]
        rect = cv2.minAreaRect(contours)
        bbox = cv2.boundingRect(contours)
        best = None
        val = 0
        for w, h in itertools.product(np.arange(-2, 1, 1),
                                      np.arange(-2, 1, 1)):
            for x, y in itertools.product(np.arange(-1, 2, 1), np.arange(-1, 2, 1)):
                c_rect = (rect[0][0] + x, rect[0][1] + y), (rect[1][0] + w, rect[1][1] + h), rect[2]
                cont = np.int0(cv2.boxPoints(c_rect)[None, :, :])
                im_ = np.zeros((self.size[0], self.size[1]), dtype=np.uint8)
                cv2.fillConvexPoly(im_, cont, 1)
                iou = ((im_ * original) > 0).sum() / (((im_ + original) > 0).sum() + 1e-07)
                if val < iou:
                    val = iou
                    best = c_rect
                    bbox = cv2.boundingRect(cont)
        # print("in", best, val)
        return (best, bbox) if val > 0.5 else (None, None)

    def crop_image(self, img, idx, augment):
        objects = []
        centroids = []
        img_masks = self.data.loc[self.data['ImageId'] == self.images[idx], 'EncodedPixels'].tolist()
        for mask in img_masks:
            if isinstance(mask, str):
                decoded = rle_decode(mask)
                objects.append(decoded)
        masks = np.stack(objects, axis=-1) if objects else None
        if self.transforms:
            transformed = self.transforms(image=img, mask=masks)
            if "mask" in transformed and transformed["mask"] is not None:
                objects = []
                for i in range(transformed["mask"].shape[-1]):
                    moments = cv2.moments(transformed["mask"][:, :, i])
                    if moments["m00"] > 16:
                        centroids.append((int(moments["m10"] / moments["m00"]), int(moments["m01"] / moments["m00"])))
                        objects.append(transformed["mask"][:, :, i])
                    # assert transformed["mask"][centroids[-1][0], centroids[-1][1], i] > 0
            img = transformed["image"]
        else:
            nobjects = []
            for obj in objects:
                moments = cv2.moments(obj)
                if moments["m00"] > 16:
                    centroids.append((int(moments["m10"] / moments["m00"]), int(moments["m01"] / moments["m00"])))
                    nobjects.append(obj)
            objects = nobjects
        if not augment:
            return img, [obj for obj in objects if obj.sum() > 0]
        nd_crop = np.array(self.crop_size)
        nd_size = np.array(self.img_size)
        if np.random.rand() > 0.3 and objects:
            choise = np.random.randint(0, len(objects) - 1) if len(objects) > 1 else 0
            center = np.clip(np.random.normal(centroids[choise], nd_crop / 18), nd_crop / 2, nd_size - nd_crop / 2 - 1)
        else:
            center = np.clip(np.random.normal(nd_size / 2, nd_size / 4), nd_crop / 2, nd_size - nd_crop / 2 - 1)
        center = center.astype(np.uint32)
        img = img[center[1] - self.crop_size[1] // 2: center[1] + self.crop_size[1] // 2,
              center[0] - self.crop_size[0] // 2: center[0] + self.crop_size[0] // 2, :].copy()
        nobjects = [obj[center[1] - self.crop_size[1] // 2: center[1] + self.crop_size[1] // 2,
                    center[0] - self.crop_size[0] // 2: center[0] + self.crop_size[0] // 2].copy()
                    for obj in objects]
        # assert (not choise) or nobjects[choise].sum() > 0, (center, centroids[choise]) if choise else None
        return img, [obj for obj in nobjects if obj.sum() > 0]

    def get_masks(self, objects):
        center_mask = np.zeros((1, self.size[0] // self.down_ratio, self.size[1] // self.down_ratio))
        seg_mask = np.zeros((1, self.size[0], self.size[1]))
        data_type = namedtuple("dims", self.data_cols.keys())
        dims = {value: [] for value in self.data_cols}
        if objects:
            for i, obj in enumerate(objects):
                original = obj
                decoded = cv2.medianBlur(cv2.dilate(cv2.erode(original,
                                                              cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))),
                                                    cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))), ksize=3)
                rect, bbox = self.abj_rect(decoded, original) if (decoded > 0).sum() > 0 else (None, None)
                if rect:
                    seg_mask += decoded
                    down_rect = ((int(rect[0][0] // self.down_ratio + 0.5), int(rect[0][1] // self.down_ratio + 0.5)),
                                 (max(3, rect[1][0] // self.down_ratio), max(3, rect[1][1] // self.down_ratio)),
                                 rect[2])
                    df = (rect[0][0] - int(down_rect[0][0]) * self.down_ratio,
                          rect[0][1] - int(down_rect[0][1]) * self.down_ratio)

                    down_bbox = [down_rect[0][0] - bbox[2] // 2, down_rect[0][1] - bbox[3] // 2,
                                 2 * (bbox[2] // 2) + 1, 2 * (bbox[3] // 2) + 1]
                    center_mask[0, :, :] = draw_gaussian(center_mask[0, :, :], down_rect, down_bbox)

                    assert center_mask[0, int(down_rect[0][1]), int(down_rect[0][0])] == 1
                    assert int(down_rect[0][0]) * self.down_ratio + df[0] == rect[0][0]
                    if self.angle_encoding == "sin":
                        angle = -4 * rect[2]
                        item = data_type(down_rect[0], rect[1], df, (np.cos(angle), np.sin(angle)))
                    else:
                        item = data_type(down_rect[0], rect[1], df, (90 + rect[2],))
                    for key, value in item._asdict().items():
                        assert isinstance(value, tuple) or isinstance(value, list)
                        dims[key].append(value)

        return center_mask, dims, seg_mask,

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        img = cv2.imread("airbus/train_v2/" + self.images[i])
        img, masks = self.crop_image(img, i, self.augment)
        center_mask, dimension_data, seg_mask = self.get_masks(masks)

        self.stat[center_mask.sum() > 1e-05] += 1

        ret = {"input": img.transpose(2, 0, 1).astype(np.float32),
               "center": center_mask.astype(np.float32)
               # "instance_seg": np.array(masks).astype(np.float32)
               # "seg": seg_mask.astype(np.float32)
               }
        if self.segmentation:
            ret.update({"instance_seg": np.array(masks).astype(np.float32)})
        for key, val in dimension_data.items():
            dimension_data[key] = np.row_stack(
                (np.array(val, dtype=np.int64 if key == "pos" else np.float32),
                 -np.ones((self.max_objects - len(val),
                           self.data_cols[key]), dtype=np.int64 if key == "pos" else np.float32)
                 )) if val else -np.ones((self.max_objects,
                                          self.data_cols[key]), dtype=np.int64 if key == "pos" else np.float32)
        ret.update(dimension_data)
        return ret
