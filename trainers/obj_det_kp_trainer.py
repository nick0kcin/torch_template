from .base_trainer import BaseTrainer
from albumentations import Flip, ShiftScaleRotate, Normalize
import cv2
import numpy as np
from torch.nn.functional import max_pool2d
import functools
from copy import copy
from torch import Tensor, atan2
from transforms.MedianBlur import MedianBlur
from torchvision import transforms

class ObjDetKPTrainer(BaseTrainer):
    def __init__(self, model, losses, loss_weights, metrics, teacher=None, optimizer=None, num_iter=-1, print_iter=-1, device=None,
                 batches_per_update=1, window_r=3, nms_thr=0.75, k=200, down_ratio=2, thr=0.1):
        super(ObjDetKPTrainer, self).__init__(model, losses, loss_weights, metrics, teacher, optimizer, num_iter, print_iter,
                                              device, batches_per_update)
        self.window_r = window_r
        self.nms_thr = nms_thr
        self.down_ratio = down_ratio
        self.K = k
        self.thr = thr

    def predict(self, dataloader, tta=None):
        return super(ObjDetKPTrainer, self)._predict(dataloader,
                                                      tta,
                                                      self.get_rects,
                                                      self.nms)

    def get_rects_batch(self, output, scale, image_id=None):
        return [self.get_rects(output, scale, i, 0 if image_id is None else image_id.item()) for i in range(output[-1]["center"].shape[0])]

    def get_rects_batch_debug(self, output, scale, image_id=None):
        return [self.get_debug_rects(output, scale, i) for i in range(output["center"].shape[0])]

    def nms_batch(self, boxes):
        return [self.nms(output) for output in boxes]

    def get_debug_rects(self, output, scale, image_id):
        w = output["wh"][image_id, :, 0]
        h = output["wh"][image_id, :, 1]
        dx = output["bias"][image_id, :, 0]
        dy = output["bias"][image_id, :, 1]
        x = output["pos"][image_id, :, 0]
        y = output["pos"][image_id, :, 1]
        angle = output["angle"][image_id, :, 0]
        cost = (x >= 0).float()
        # assert output["center"][0, x, y] == 1
        result = []
        costs, idx = cost.topk(min(self.K, cost.shape[0]))
        num_good = (costs > self.thr).sum()
        idx = idx[:num_good]
        for i in idx:
            if "angle" in output:
                rect = (
                    (x[i].item() * self.down_ratio + dx[i].item() * scale,
                     y[i].item() * self.down_ratio + dy[i].item() * scale),
                    (w[i].item() * scale,
                     h[i].item() * scale),
                    -np.rad2deg(np.arcsin(angle[i].item()))
                )
                # print("out", rect)
            else:
                rect = (int(x[i].item() * self.down_ratio - h[i].item() * self.down_ratio) * scale,
                        int(y[i].item() * self.down_ratio - w[i].item() * self.down_ratio) * scale,
                        int(2 * h[i].item()) * scale * self.down_ratio, int(2 * w[i].item()) * scale * self.down_ratio)
            anno = output["center"][image_id, :, x[i], y[i]].argmax().item()
            result.append(
                {"image_id": image_id, "bbox": rect, "category_id": anno, "score": cost[i].item()})
        return result

    def get_rects(self, output, scale, image_id, _id):
        any_class_map = output[0]["center"].max(1, keepdim=True)[0]
        radius = max(1, int(self.window_r / scale + 0.5))
        extrema_map = (max_pool2d(any_class_map, 2 * radius + 1, stride=1) ==
                       any_class_map[:, :, radius:-radius, radius:-radius]).cpu()
        points = extrema_map.nonzero()
        points[:, 2:] += radius
        x = points[:, 3]
        y = points[:, 2]
        smooth_wh = output[0]['wh']
        w = smooth_wh[image_id, 0, points[:, 2], points[:, 3]]
        h = smooth_wh[image_id, 1, points[:, 2], points[:, 3]]
        dx = output[0]["bias"][image_id, 0, points[:, 2], points[:, 3]]
        dy = output[0]["bias"][image_id, 1, points[:, 2], points[:, 3]]

        if "angle" in output:
            # smooth_angle = output['angle']
            # angle = torch.atan2(smooth_angle[image_id, 1, points[:, 2], points[:, 3]],
            #                     smooth_angle[image_id, 0, points[:, 2], points[:, 3]]) / 2
            bin_size = 90 / output["angle"].shape[1]
            smooth_angle = MedianBlur((2 * radius + 1, 2 * radius + 1))(output["angle"].argmax(1, keepdim=True).float())
            angle = smooth_angle.float() * bin_size + bin_size / 2 + output['anglediff'].tanh() * bin_size / 2
            angle = angle[image_id, 0, points[:, 2], points[:, 3]]

        cost = any_class_map[image_id, points[:, 1], points[:, 2], points[:, 3]]
        result = []
        # for img in range(cost.shape[0]):
        costs, idx = cost.topk(min(self.K, cost.shape[0]))
        num_good = (costs.sigmoid() > self.thr).sum()
        idx = idx[:num_good]
        for i in idx:
            if "angle" in output:
                rect = (
                          (x[i].item() * self.down_ratio + dx[i].item() * scale,
                           y[i].item() * self.down_ratio + dy[i].item() * scale),
                          (w[i].item() * scale,
                           h[i].item() * scale),
                           angle[i].item() - 90
               )
                # print("out", rect)
            else:
                rect = (int(points[i, 3].item() * self.down_ratio - w[i].item() / 2 + dy[i].item()) * scale,
                        int(points[i, 2].item() * self.down_ratio - h[i].item() / 2 + dx[i].item()) * scale,
                        int(w[i].item()) * scale, int(h[i].item()) * scale)
            anno = output[0]["center"][image_id, :, points[i, 2], points[i, 3]].argmax().item()
            result.append({"image_id": _id, "bbox": rect, "category_id": anno, "score": cost[i].sigmoid().item()})
        return result

    def nms(self, boxes):
        boxes = list(sorted(boxes, key=functools.cmp_to_key(lambda x, y: y["score"] - x["score"])))
        for i, box in enumerate(boxes):
            rect = box["bbox"]
            eta_box = copy(rect)
            for box2 in boxes[i + 1:]:

                rect2 = box2["bbox"]
                assert len(rect) == len(rect2)
                assert len(rect) == 4 or len(rect) == 3
                if len(rect) == 3:
                    intersect = cv2.rotatedRectangleIntersection(rect, rect2)[1]
                    intersection_area = cv2.contourArea(intersect) if intersect is not None else 0
                    union_area = rect[1][0] * rect[1][1] + rect2[1][0] * rect2[1][1] - intersection_area
                    iou = intersection_area / (union_area + 1e-07)
                else:
                    x_left = max(rect[0], rect2[0])
                    y_top = max(rect[1], rect2[1])
                    x_right = min(rect[0] + rect[2], rect2[0] + rect2[2])
                    y_bottom = min(rect[1] + rect[3], rect2[1] + rect2[3])
                    if x_right < x_left or y_bottom < y_top:
                        intersection_area = 0
                    else:
                        intersection_area = (x_right - x_left) * (y_bottom - y_top)
                    iou = intersection_area / (
                            rect[2] * rect[3] + rect2[3] * rect2[2] - intersection_area)

                if iou > self.nms_thr:
                    boxes.remove(box2)
                    # eta_box[0] = eta_box[0]
        return boxes

    def boxes_to_segmap(self, boxes, shape):
        boxes.sort(key=lambda x: -x["bbox"][1][0] * x["bbox"][1][1])
        mask = np.zeros((len(boxes), shape[0], shape[1]), dtype=np.uint8)
        for i, box in enumerate(boxes):
            if len(box["bbox"]) == 3:
                points = cv2.boxPoints(box["bbox"])[None, :, :]
                cv2.fillConvexPoly(mask[i,:,:], points.astype(np.int32),
                                   1.0)
            # if len(boxes) == 1:
            #     cv2.imshow("mask", mask[0,:,:]*255)
            #     cv2.waitKey()
        # channels = [(mask == i).astype(np.float32) for i in range(1, len(boxes) + 1)] if boxes \
        #     else mask.astype(np.float32)[None, :, :]
        # print([chan.sum() for chan in channels])
        return Tensor(mask)

    def boxes_to_segmap_batch(self, batch_boxes, shape):
        return [self.boxes_to_segmap(boxes, shape).to(device=self.device, non_blocking=True) for boxes in batch_boxes]

    def test(self, dataloader, verbose=0, tta=None):
        return super()._test(dataloader, "instance_seg", tta,
                             mapper=self.get_rects_batch,
                             # aggregator=lambda x: self.boxes_to_segmap_batch(x,
                             #                                           len(dataloader.dataset)),
                             verbose=verbose
                             )

    def debug(self, dataloader):
        return self.metric_handler(super(ObjDetKPTrainer, self)._debug_run(dataloader,
                              mapper=self.get_rects_batch_debug,
                              label_mapper=lambda x: x["instance_seg"],
                              aggregator=lambda x: self.boxes_to_segmap_batch(x,
                                                                             dataloader.dataset.size)
                             ))
    def visualize(self, dataloader):
        predictor = super()._predict(dataloader, tta=None,
                                     mapper=lambda batch,i,j: (self.get_rects_batch(batch,i,j), batch[0]["center"]),
                                     label_mapper=lambda x: x,
                                     aggregator=lambda x: (self.nms_batch(x[0]), x[1]))
        cv2.startWindowThread()
        inv_normalize = Normalize(
            mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
            std=[1 / 0.229, 1 / 0.224, 1 / 0.255], max_pixel_value=1
        )
        for i, (img, (rects, map), gt) in enumerate(predictor):
            # loss = self.loss["center"](map, gt)
            loss = 1
            # print(gt["meta"], loss)
            if loss > 0:
                im = inv_normalize(image=img.numpy()[0, :, :, :].transpose(1, 2, 0))["image"] * 255
                img = im.astype(np.uint8)
                mp = map.sigmoid().cpu().numpy()[0, :, :, :].max(0)
                mp = cv2.resize(mp, (4 * mp.shape[1], 4 * mp.shape[0]))
                # gt_ = gt["center"].cpu().numpy()[0, :, :, :].max(0)
                # gt_ = cv2.resize(gt_, (1024, 1024))
                im = np.zeros(img.shape, dtype=np.uint8)
                colors = [
                    (255, 0, 0),  # plane
                    (0, 255, 0),  # car
                    (0, 0, 255),  # truck
                    (255, 0, 255),  # railway pink
                    (0, 255, 255),  # yellow submarine
                    (255, 255, 0),  # buildings
                    (255, 255, 255),  # helipad
                    (128, 255, 128),  # container
                    (128, 0, 255)  # tower
                ]
                names = [
                    "plane",
                    "car",
                    "truck",
                    "train",
                    "ship",
                    "build",
                    "H",
                    "box",
                    "tower"
                ]
                for rect in rects[0]:
                    # contour = cv2.boxPoints(rect["bbox"])
                    # cv2.drawContours(im, [np.int0(contour)], 0, (255, 0, 255), 1)
                    cv2.rectangle(im, (rect["bbox"][0], rect["bbox"][1]),
                                      (rect["bbox"][0] + rect["bbox"][2],
                                       rect["bbox"][1] + rect["bbox"][3]),
                                  colors[rect["category_id"]], thickness=2)

                    c1, c2 = (rect["bbox"][0], rect["bbox"][1]),\
                             (rect["bbox"][0] + rect["bbox"][2], rect["bbox"][1] + rect["bbox"][3])
                    tf = 1  # font thickness
                    t_size = cv2.getTextSize(names[rect["category_id"]], 0, fontScale=2 / 3, thickness=tf)[0]
                    c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
                    # cv2.rectangle(img, c1, c2, colors[rect["category_id"]], -1, cv2.LINE_AA)  # filled
                    cv2.putText(im, names[rect["category_id"]], (c1[0], c1[1] - 2), 0, 2 / 3, [225, 255, 255],
                                thickness=tf,
                                lineType=cv2.LINE_AA)
                # gt_rects = gt["meta"]
                # for rect in gt_rects:
                #     cv2.rectangle(im, (rect["bbox"][0].item(), rect["bbox"][1].item()),
                #                   (rect["bbox"][0].item() + rect["bbox"][2].item(),
                #                    rect["bbox"][1].item() + rect["bbox"][3].item()),
                #                   tuple([i + 200 if i < 255 else i - 100 for i in colors[rect["category_id"].item()]]),
                #                   thickness=1, lineType=cv2.LINE_4)
                # cv2.imshow("res", im | img)
                # cv2.imshow("mp", mp)
                # cv2.imshow("gt", gt_)
                # cv2.imshow("img", img)
                # cv2.waitKey(10)
                return im, mp

