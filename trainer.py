import torch
from torch.nn import DataParallel
from tqdm import tqdm
import sys
from torch.nn.functional import max_pool2d
import numpy as np
import cv2
import functools
try:
    from apex import amp
    APEX = True
except ModuleNotFoundError:
    APEX = False


class ModelWithLoss(torch.nn.Module):
    def __init__(self, model, loss, loss_weights):
        super(ModelWithLoss, self).__init__()
        self.model = model
        self.loss = loss
        self.loss_weights = loss_weights

    def forward(self, batch):
        outputs = self.model(batch['input'])
        loss_value = 0
        loss_stats = {key: 0.0 for key in self.loss}
        if isinstance(outputs[-1], dict):
            for key, loss in self.loss.items():
                loss_stats[key] = loss(outputs[-1], batch)
            outputs = outputs[-1]
        else:
            key = list(self.loss.keys())[0]
            loss_stats[key] = list(self.loss.values())[0](outputs, batch[key])

        loss_value += sum({key: self.loss_weights[key] * val for key, val in loss_stats.items()}.values())
        return outputs, loss_value, loss_stats


class Trainer(object):
    def __init__(self, model, losses, loss_weights,  optimizer=None, num_iter=-1, print_iter=-1, device=None,
                 batches_per_update=1, k=20, thr=0.1, window_r=2):
        self.num_iter = num_iter
        self.print_iter = print_iter
        self.device = device
        self.K = k
        self.thr = thr
        self.window_r = window_r
        self.batches_per_update = batches_per_update
        self.optimizer = optimizer
        self.loss = losses
        self.loss_weights = loss_weights
        self.model_with_loss = ModelWithLoss(model, self.loss, self.loss_weights)

    def set_device(self, gpus, device):
        if len(gpus) > 1:
            self.model_with_loss = DataParallel(
                self.model_with_loss, device_ids=gpus).to(device)
        else:
            self.model_with_loss = self.model_with_loss.to(device)

        for state in self.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device=device, non_blocking=True)

    def run_epoch(self, phase, epoch, data_loader, require_predict):
        debug = False
        model_with_loss = self.model_with_loss
        if phase == 'train':
            model_with_loss.train()
        else:
            #model_with_loss = self.model_with_loss.module
            model_with_loss.eval()
            torch.cuda.empty_cache()

        results = dict() if phase != "test" else []
        predicts = []
        current_predict = []
        moving_loss = 0
        num_iters = len(data_loader) if (self.num_iter < 0) or (phase != "train") else self.num_iter
        try:
            scales = data_loader.dataset.scales
        except:
            scales = (1, )
        with tqdm(data_loader, total=num_iters, file=sys.stdout) as bar_object:
            for iter_id, batch in enumerate(bar_object):

                if iter_id >= num_iters:
                    break
                for k in batch:
                    if k != 'meta':
                        batch[k] = batch[k].to(device=self.device, non_blocking=True)
                if phase == 'train':
                    output, loss, loss_stats = model_with_loss(batch)
                    loss = loss.mean() / self.batches_per_update
                elif phase == "val":
                    with torch.no_grad():
                        output, loss, loss_stats = model_with_loss(batch)
                        loss = loss.mean() / self.batches_per_update
                        if require_predict:
                            rectangles = self.predict(output, image_id=iter_id // len(scales), down_ratio=4,
                                                      scale=scales[iter_id % len(scales)])
                            current_predict.extend(rectangles)
                            if not ((iter_id+1) % len(scales)):
                                current_predict = self.nms(current_predict)
                                predicts.extend(current_predict)
                                if debug:
                                    cv2.startWindowThread()
                                    img = (batch['input'][0, :, :, :].cpu().numpy().transpose(1, 2, 0) * 255).astype(
                                        np.uint8).copy()
                                    self.visualize(img, current_predict)
                                current_predict = []
                else:
                    with torch.no_grad():
                        output = model_with_loss.model(batch['input'])[-1]
                        rectangles = self.predict(output, image_id=iter_id // len(scales), down_ratio=4,
                                                  scale=scales[iter_id % len(scales)])
                        current_predict.extend(rectangles)
                        if not ((iter_id + 1) % len(scales)):
                            current_predict = self.nms(current_predict)
                            predicts.extend(current_predict)
                            if debug:
                                cv2.startWindowThread()
                                img = (batch['input'][0, :, :, :].cpu().numpy().transpose(1, 2, 0) * 255).astype(
                                    np.uint8).copy()
                                self.visualize(img, current_predict, output["center"], batch["meta"])
                            current_predict = []

                if phase == 'train':
                    if iter_id % self.batches_per_update == 0:
                        self.optimizer.zero_grad()
                    # if APEX:
                    #     with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    #         scaled_loss.backward()
                    loss.backward()
                    if (iter_id + 1) % self.batches_per_update == 0:
                        self.optimizer.step()
                if phase != "test":
                    moving_loss += loss.item()
                    results = {key: results.get(key, 0) + val.mean().item() for (key, val) in loss_stats.items()}
                    del loss, loss_stats
                    bar_object.set_postfix_str("{phase}:[{epoch}]' loss={loss} {losses}".
                                               format(phase=phase, epoch=epoch,
                                                      loss=moving_loss * self.batches_per_update / (iter_id + 1),
                                                      losses={k: v / (iter_id + 1)
                                                              for k, v in results.items()}))
                else:
                    bar_object.set_postfix_str("{phase}:[{epoch}]".
                                               format(phase=phase, epoch=epoch))
                bar_object.update(1)
                if self.print_iter > 0 and not (iter_id % self.print_iter):
                    bar_object.write(bar_object.__str__())
                del output
        if phase != "test":
            results = {k: v / num_iters for k, v in results.items()}
            results.update({'loss': moving_loss / num_iters})
            return (results, predicts) if require_predict else results
        return predicts

    def val(self, epoch, data_loader, require_predict=False):
        return self.run_epoch('val', epoch, data_loader, require_predict)

    def train(self, epoch, data_loader):
        return self.run_epoch('train', epoch, data_loader, require_predict=False)

    def test(self, epoch, data_loader):
        return self.run_epoch("test", epoch, data_loader, require_predict=True)

    def predict(self, output, image_id, down_ratio=4, scale=1):
        any_class_map = output["center"].max(1, keepdim=True)[0]
        radius = max(1, int(self.window_r / scale + 0.5))
        extrema_map = (max_pool2d(any_class_map, 2 * radius + 1, stride=1) ==
                       any_class_map[:, :, radius:-radius, radius:-radius]).cpu()
        points = extrema_map.nonzero()

        smooth_wh = (max_pool2d(output['dim'], 2 * radius + 1, stride=1)).cpu()
        w = smooth_wh[0, 0, points[:, 2], points[:, 3]]
        h = smooth_wh[0, 1, points[:, 2], points[:, 3]]

        points[:, 2:] += radius
        cost = any_class_map[0, points[:, 1], points[:, 2], points[:, 3]]
        costs, idx = cost.topk(min(self.K, cost.shape[0]))
        num_good = (costs.sigmoid() > self.thr).sum()
        idx = idx[:num_good]
        result = []
        for i in idx:
            rect = (int(points[i, 2].item() * down_ratio - h[i].item() * down_ratio) * scale,
                    int(points[i, 3].item() * down_ratio - w[i].item() * down_ratio) * scale,
                    int(2 * h[i].item()) * scale * down_ratio, int(2 * w[i].item()) * scale * down_ratio)
            anno = output["center"][0, :, points[i, 2], points[i, 3]].argmax().item()
            result.append({"image_id": image_id, "bbox": rect, "category_id": anno, "score": cost[i].sigmoid().item()})
        return result

    @staticmethod
    def nms(boxes, thr=0.75):
        boxes = list(sorted(boxes, key=functools.cmp_to_key(lambda x, y: y["score"]-x["score"])))
        for i, box in enumerate(boxes):
            rect = np.array(box["bbox"])
            eta_box = rect.copy()
            for box2 in boxes[i+1:]:

                rect2 = np.array(box2["bbox"])
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
                if iou > thr:
                    boxes.remove(box2)
                    eta_box[0] = eta_box[0]
        return boxes

    @staticmethod
    def visualize(img, boxes, center=None, meta=None):
        class_colors = [(255, 0, 0), (255, 255, 0), (255, 255, 255), (255, 0, 255), (0, 255, 255), (0, 0, 0)]
        class_names = ["boat", "buoy", "vessel", "military", "ice", "other"]
        for box in boxes:
            rect = box["bbox"]
            anno = box["category_id"]
            cv2.rectangle(img, (rect[1], rect[0]), (rect[3] + rect[1], rect[2] + rect[0]),
                          class_colors[anno], 2)
            cv2.putText(img, class_names[anno], (rect[1], rect[0] + rect[2]), cv2.FONT_HERSHEY_PLAIN, 2,
                        class_colors[anno], 3)
        cv2.imshow("123", img)
        center_img = cv2.resize(center.max(dim=1)[0].sigmoid().cpu().numpy()[0, :, :], (img.shape[1], img.shape[0]))
        cv2.imshow("center", center_img)
        key = cv2.waitKey()
        if key == ord("s"):
            cv2.imwrite("/opt/project/exp/images/" + meta["name"][0].replace("/", "_"), img)
            cv2.imwrite("/opt/project/exp/images/center_" + meta["name"][0].replace("/", "_"),
                        (center_img*255).astype(np.uint8))
