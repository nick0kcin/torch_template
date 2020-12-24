import json
import os
from collections import Counter
from collections import namedtuple
from shutil import copy2

from pycocotools.coco import COCO
from sklearn.cluster import KMeans
from torch.utils.data import Dataset

from image_utils import *


class Xview(Dataset):
    def __init__(self, path, label_path, classes_file, crop=True, transforms=None):
        cv2.startWindowThread()
        self.path = path
        self.label_path = label_path
        self.gen_labels(label_path, classes_file)
        self.down_ratio = 4
        self.data_cols = {"pos": 2, "wh": 2, "bias": 2, "cat": 1}
        self.max_objects = max([len(v[1]) for v in self.images])
        self.transforms = transforms
        self.crop = crop
        self.Coco = None

    def coco(self, super_class=False):
        if not self.Coco:
            images = [{"file_name": row[0], "id": i} for i, row in enumerate(self.images)]
            categories = [{"id": value, "name": key} for key, value in self.super_cat_idx.items()]
            annotations = []
            anno_id = 0
            for i, row in enumerate(self.images):
                anno = row[1]
                for rect in anno:
                    bbox = rect["bbox"]
                    bbox = [int(bbox[0]), int(bbox[1]), int(bbox[2] - bbox[0]), int(bbox[3] - bbox[1])]
                    if bbox[2] * bbox[3] >= 100:
                        category_id = self.super_cat_idx[rect["super_label"]]
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

    def txt_file(self, name):
        with open(name, "w") as f:
            for i, row in enumerate(self.images):
                f.write(f"{name.split('.')[0]}_cut/" + row[0] + " ")
                for rect in row[1]:
                    bbox = rect["bbox"]
                    category_id = self.super_cat_idx[rect["super_label"]]
                    f.write(f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]},{category_id} ")
                f.write("\n")

    def txt_files(self, name):
        shape = cv2.imread(os.path.join(self.path, self.images[0][0])).shape
        for i, row in enumerate(self.images):
            try:
                from shutil import copy2
                copy2(os.path.join(self.path, row[0]), os.path.join(self.path, row[0]).replace("/images", "/imgs"))
                os.remove(os.path.join(self.path, row[0]).replace(".tif", "_0.tif"))
            except FileNotFoundError:
                pass
            if True:
                with open(os.path.join(self.path, row[0].replace(".tif", ".txt")).replace("images/", "labels/"),
                          "w") as f:
                    for rect in row[1]:
                        bbox = rect["bbox"]
                        category_id = self.super_cat_idx[rect["super_label"]]
                        bbox = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2,
                                (bbox[2] - bbox[0]), (bbox[3] - bbox[1])]
                        f.write(
                            f"{category_id} {bbox[0] / shape[0]} {bbox[1] / shape[1]} {bbox[2] / shape[0]} {bbox[3] / shape[1]}\n")
            else:
                print("empty image")

    def compute_anchors(self):
        rects = []
        for i, row in enumerate(self.images):
            for rect in row[1]:
                bbox = rect["bbox"]
                rects.append((bbox[2] - bbox[0], bbox[3] - bbox[1]))
        clusters = KMeans(n_clusters=9).fit(rects)
        print(clusters.cluster_centers_)

    def gen_labels(self, label_path, classes_file, mapping=True):
        # self.objects = []
        # self.labels = Counter()
        self.labels_map = {}
        # self.strips = Counter()
        # self.strip_dict = {}
        # self.strip_counters = {}
        self.super_cats = {}
        # self.super_labels = Counter()
        self.cat_iera = {}
        self.images = {}
        self.raw_labels = {}
        with open(classes_file) as f:
            for line in f:
                split1 = line.partition(":")
                key = split1[0].strip()
                value = split1[2].split("//")[mapping].strip()
                super_cat = split1[2].split("//")[-1].strip()
                self.labels_map.update({int(key): value})
                self.super_cats.update({int(key): super_cat})
                self.cat_iera.update({super_cat: self.cat_iera.get(super_cat, set()) | {value}})
        self.super_cat_idx = {s_cat: i for i, s_cat in enumerate(self.cat_iera)}
        self.cat_idx = {}
        for s in self.cat_iera.values():
            self.cat_idx.update({cat: i for i, cat in enumerate(s)})
        with open(label_path) as f:
            data = json.load(f)
        features = data["features"]
        avaible_files = set(os.listdir(self.path))
        for item in features:
            img_data = item["properties"]
            label = self.labels_map.get(img_data["type_id"], "Other")
            super_label = self.super_cats.get(img_data["type_id"], "Other")
            if img_data["image_id"] in avaible_files and label != "Other":
                # self.objects.append({
                patch = {
                    "image": img_data["image_id"],
                    "bbox": list(map(int, img_data["bounds_imcoords"].split(","))),
                    "label": label,
                    "super_label": super_label,
                    "strip": img_data["cat_id"]
                }
                # })
                self.images.update({img_data["image_id"]:
                                        self.images.get(img_data["image_id"], []) + [patch]})
                self.raw_labels.update({img_data["image_id"]:
                                            self.raw_labels.get(img_data["image_id"], []) + [img_data]})

        self.images = list(self.images.items())
        self.raw_labels = list(self.raw_labels.items())

    def cut_big_image(self, img, objects, name, size=1024, stride=0.7):
        last = None
        new_labels = []
        for i, (sx, sy) in enumerate(
                itertools.product(range(size, img.shape[0] + size - 1, int(size * stride)),
                                  range(size, img.shape[1] + size - 1, int(size * stride)))):
            x, y = min(sx, img.shape[1]), min(sy, img.shape[0])
            nx, ny = max(x - size, 0), max(y - size, 0)
            if last is None:
                last = np.array([x, y, nx, ny])
            elif np.abs(np.array([x, y, nx, ny]) - last).mean() < 20:
                last = np.array([x, y, nx, ny])
                continue
            last = np.array([x, y, nx, ny])
            crop = img[ny: y, nx: x, :].copy()
            vertex = [nx, ny]
            crop_rect = [nx, ny, x, y]
            good_objects = filter(
                lambda it: area(intersection(list(map(int, it["bounds_imcoords"].split(","))),
                                             crop_rect)) > 0.5 * area(
                    list(map(int, it["bounds_imcoords"].split(",")))) > 0,
                objects)
            obj = [{"properties":
                {
                    "image_id": it["image_id"].replace(".tif", f"_{i}.tif"),
                    "cat_id": it["cat_id"],
                    "type_id": it["type_id"],
                    "bounds_imcoords": ",".join(
                        map(str, move(
                            intersection(list(map(int, it["bounds_imcoords"].split(","))), crop_rect),
                            vertex, crop.shape)))
                }
            } for it in good_objects]
            new_labels.extend(obj)
            assert crop.shape == (min(img.shape[0], size), min(img.shape[1], size), 3)
            cv2.imwrite(name.replace("images", "cut").replace(".tif", f"_{i}.tif"), crop)
        return new_labels

    def cut_images(self):
        features = {"features": []}
        for item, (name, label) in enumerate(self.raw_labels):
            full_name = os.path.join(self.path, name)
            img = cv2.imread(full_name)
            objects = label
            data = self.cut_big_image(img, objects, full_name)
            features["features"].extend(data)
        with open(self.label_path.replace("xView", "xViewCut"), "w") as f:
            f.write(json.dumps(features))

    def __len__(self):
        return len(self.images)

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
            cv2.rectangle(img, tuple(np.int32(obj[:2])), tuple(np.int32(obj[2:])), (255, 0, 0))
            # points = cv2.boxPoints(obj)
            # cv2.drawContours(cv2.UMat(img), np.int0(points[None, :, :]), 0, (255, 0, 0))
        cv2.imshow(name, img)
        cv2.waitKey()

    def debug_big(self, cat_id):
        searched = filter(lambda x: x["label"] == self.labels_map[cat_id], self.objects)
        s = list(searched)
        sizes = [np.clip(np.sqrt((x["bbox"][2] - x["bbox"][0]) ** 2 + (x["bbox"][3] - x["bbox"][1]) ** 2), 0, 200) for x
                 in s]
        print(np.mean(sizes), 3 * np.std(sizes))
        _, kmeans, _ = cv2.kmeans(np.array([sizes]).transpose(1, 0).astype(np.float32),
                                  2, None,
                                  (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 1000, 0.1),
                                  1000, cv2.KMEANS_RANDOM_CENTERS)
        f1 = []
        f2 = []
        for label, data in zip(kmeans[:, 0], s):
            if label:
                f1.append(data)
            else:
                f2.append(data)
        for el, el2 in itertools.zip_longest(f1, f2, fillvalue=None):
            if el:
                print(el)
                full_name = os.path.join(self.path, el["image"])
                img = cv2.imread(full_name)
                bbox = el["bbox"]
                cv2.rectangle(img, tuple(bbox[:2]), tuple(bbox[2:]), (0, 0, 255), 2)
                c = (bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2
                crop = img[np.clip(c[1] - 512, 0, img.shape[0]):
                           np.clip(c[1] + 512, 0, img.shape[0]),
                       np.clip(c[0] - 512, 0, img.shape[1]):
                       np.clip(c[0] + 512, 0, img.shape[1])].copy()
                cv2.imshow("img", crop)
            if el2:
                print(el2)
                full_name = os.path.join(self.path, el2["image"])
                img = cv2.imread(full_name)
                bbox = el2["bbox"]
                cv2.rectangle(img, tuple(bbox[:2]), tuple(bbox[2:]), (0, 255, 0), 2)
                c = (bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2
                crop = img[np.clip(c[1] - 512, 0, img.shape[0]):
                           np.clip(c[1] + 512, 0, img.shape[0]),
                       np.clip(c[0] - 512, 0, img.shape[1]):
                       np.clip(c[0] + 512, 0, img.shape[1])].copy()
                cv2.imshow("img2", crop)
            cv2.waitKey()

    def get_masks(self, objects, labels, img_size):
        center_mask = np.zeros(
            (len(self.super_cat_idx), img_size[0] // self.down_ratio, img_size[1] // self.down_ratio))
        data_type = namedtuple("dims", self.data_cols.keys())
        dims = {value: [] for value in self.data_cols}
        assert len(objects) == len(labels)
        for box, cls in zip(objects, labels):
            rect = ((box[0] + box[2]) // 2, (box[1] + box[3]) // 2), (box[2] - box[0], box[3] - box[1]), 0
            bbox = box[0], box[1], box[2] - box[0], box[3] - box[1]
            down_rect = ((int(rect[0][0] // self.down_ratio + 0.5), int(rect[0][1] // self.down_ratio + 0.5)),
                         (np.clip(rect[1][0] // self.down_ratio, 3, 50),
                          np.clip(rect[1][1] // self.down_ratio, 3, 50)),
                         rect[2])
            df = (rect[0][0] - int(down_rect[0][0]) * self.down_ratio,
                  rect[0][1] - int(down_rect[0][1]) * self.down_ratio)

            dside = (np.clip(int(bbox[2] // (2 * self.down_ratio)), 3, 100),
                     np.clip(int(bbox[3] // (2 * self.down_ratio)), 3, 100))
            down_bbox = [down_rect[0][0] - dside[0], down_rect[0][1] - dside[1],
                         2 * dside[0] + 1,
                         2 * dside[1] + 1]
            cls_idx = self.super_cat_idx[cls[0]]
            # print(down_bbox[2:])
            assert down_bbox[3] >= 3 and down_bbox[2] >= 3
            center_mask[cls_idx, :, :] = draw_gaussian(center_mask[cls_idx, :, :], down_rect, down_bbox)

            # print(center_mask[cls_idx, int(down_rect[0][1]), int(down_rect[0][0])], down_rect, down_bbox, box)

            assert center_mask[cls_idx, int(down_rect[0][1]), int(down_rect[0][0])] == 1
            assert int(down_rect[0][0]) * self.down_ratio + df[0] == rect[0][0]
            assert int(down_rect[0][1]) * self.down_ratio + df[1] == rect[0][1]

            item = data_type(down_rect[0], rect[1], df, (self.cat_idx[cls[1]],))

            for key, value in item._asdict().items():
                assert isinstance(value, tuple) or isinstance(value, list)
                dims[key].append(value)

        return center_mask, dims

    def __getitem__(self, item):
        full_name = os.path.join(self.path, self.images[item][0])
        # print(full_name)
        img = cv2.imread(full_name)
        objects = self.images[item][1]
        annotations = {"image": img, "bboxes": [obj["bbox"] for obj in objects],
                       "super_label": [(obj["super_label"], obj["label"]) for obj in objects],
                       # "label": [obj["label"] for obj in objects]
                       }

        transformed = self.transforms(**annotations)
        img_ = transformed["image"]
        bboxes = transformed["bboxes"]
        labels = transformed["super_label"]
        center_mask, dimension_data = self.get_masks(bboxes, labels, img_.shape[:2])

        ret = {
            "input": img_.transpose(2, 0, 1).astype(np.float32),
            "center": center_mask.astype(np.float32),
            "meta": item,
            "meta_name": full_name
            # "meta": [el for el in self.Coco.anns.values() if el["image_id"] == item]
        }
        for key, val in dimension_data.items():
            dimension_data[key] = np.row_stack(
                (np.array(val, dtype=np.int64 if key in {"pos", "cat"} else np.float32),
                 -np.ones((self.max_objects - len(val),
                           self.data_cols[key]), dtype=np.int64 if key in {"pos", "cat"} else np.float32)
                 )) if val else -np.ones((self.max_objects,
                                          self.data_cols[key]), dtype=np.int64 if key in {"pos", "cat"} else np.float32)
        ret.update(dimension_data)

        return ret
