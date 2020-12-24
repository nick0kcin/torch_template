import sys
import numpy as np
import torch
from torch.nn import DataParallel
from tqdm import tqdm

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
        outputs = {}
        b = len(batch["input"].shape) == 5
        for key in batch:
            if b and not isinstance(batch[key], list):
                batch[key] = batch[key].view((-1, *batch[key].shape[2:]))
            if key.startswith("input"):
                out = self.model(batch[key])
                outputs.update({k + key[5:]: val for k, val in out[-1].items()})
        loss_value = 0.0
        loss_stats = {key: -1 for key in self.loss}
        if isinstance(outputs, dict):
            for key, loss in self.loss.items():
                # if self.loss_weights[key]:
                # if key in outputs:
                try:
                    loss_stats[key] = loss(outputs, batch)
                except KeyError:
                    pass
        else:
            key = list(self.loss.keys())[0]
            loss_stats[key] = list(self.loss.values())[0](outputs, batch[key])

        loss_value += sum({key: self.loss_weights[key] * val
                           for key, val in loss_stats.items() if not isinstance(val, int) or val > 0}.values())
        return outputs, loss_value, {key: val for key, val in loss_stats.items() if not isinstance(val, int) or val > 0}


class DistillModelWithLoss(torch.nn.Module):
    def __init__(self, model, teacher, loss, loss_weights, distill_weight=1):
        super(DistillModelWithLoss, self).__init__()
        self.model = model
        self.teacher = teacher
        self.loss = loss
        self.loss_weights = loss_weights
        self.distill_weight = distill_weight

    def forward(self, batch):
        outputs = self.model(batch['input'])
        with torch.no_grad():
            lessons = self.teacher(batch['input'])
        for key in lessons[-1]:
            if key not in batch and key != "input":
                batch.update({key: lessons[-1][key]})
        loss_value = 0
        loss_stats = {key: 0.0 for key in self.loss}
        if isinstance(outputs[-1], dict):
            for key, loss in self.loss.items():
                loss_stats[key] = loss(outputs[-1], batch)
        else:
            key = list(self.loss.keys())[0]
            loss_stats[key] = list(self.loss.values())[0](outputs, batch[key])

        distill_loss_stats = {f"d_{key}": 0.0 for key in self.loss if
                              not key.startswith("d_") and batch[key].shape == outputs[0][key].shape}
        if isinstance(outputs[-1], dict):
            for key, loss in self.loss.items():
                if not key.startswith("d_") and batch[key].shape == outputs[0][key].shape:
                    distill_loss_stats[f"d_{key}"] = loss(outputs[-1], lessons[-1])
            outputs = outputs[-1]
        else:
            key = list(self.loss.keys())[0]
            distill_loss_stats[key] = list(self.loss.values())[0](outputs, lessons)

        loss_stats.update(distill_loss_stats)
        loss_value += sum({key: self.loss_weights[key[2:] if key not in self.loss else key] * val
                           for key, val in loss_stats.items() if not torch.isnan(val)}.values())
        return outputs, loss_value, loss_stats


class BaseTrainer(object):
    def __init__(self, model, losses, loss_weights, metrics, teacher=None, optimizer=None, num_iter=-1, print_iter=-1,
                 device=None,
                 batches_per_update=1):
        self.num_iter = num_iter
        self.print_iter = print_iter
        self.device = device
        self.batches_per_update = batches_per_update
        self.optimizer = optimizer
        self.loss = losses
        self.loss_weights = loss_weights
        self.metric_handler = metrics
        self.model_with_loss = ModelWithLoss(model, self.loss, self.loss_weights) if teacher is None \
            else DistillModelWithLoss(model, teacher, self.loss, self.loss_weights)

    def set_device(self, gpus, device):
        if len(gpus) > 1:
            self.model_with_loss = DataParallel(
                self.model_with_loss, device_ids=gpus, output_device="cuda:1").to(device)
        else:
            self.model_with_loss = self.model_with_loss.to(device)

        if self.optimizer:
            for state in self.optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(device=device, non_blocking=True)

    def to(self, batch):
        if not isinstance(batch, dict):
            batch[0] = batch[0].to(device=self.device, non_blocking=True)
            return batch
        for k in batch:
            if "meta" not in k:
                if isinstance(batch[k], list):
                    for i, el in enumerate(batch[k]):
                        if not isinstance(batch[k][i], str):
                            batch[k][i] = batch[k][i].to(device=self.device, non_blocking=True)
                else:
                    batch[k] = batch[k].to(device=self.device, non_blocking=True)
        return batch

    def run_epoch(self, phase, epoch, data_loader):
        model_with_loss = self.model_with_loss


        if phase == 'train':
            model_with_loss.train()
        else:
            model_with_loss.eval()
            torch.cuda.empty_cache()

        results = dict()
        moving_loss = 0
        num_iters = len(data_loader) if self.num_iter < 0 else self.num_iter
        with tqdm(data_loader, file=sys.stdout) as bar_object:
            for iter_id, batch in enumerate(bar_object):

                if iter_id >= num_iters:
                    break
                batch = self.to(batch)
                if phase == 'train':
                    output, loss, loss_stats = model_with_loss(batch)
                    loss = loss.sum(-1)
                    loss = loss.mean() / self.batches_per_update
                else:
                    with torch.no_grad():
                        output, loss, loss_stats = model_with_loss(batch)
                        loss = loss.sum(-1)
                        loss = loss.mean() / self.batches_per_update

                if phase == 'train':
                    if iter_id % self.batches_per_update == 0:
                        self.optimizer.zero_grad()
                    loss.backward()
                    if (iter_id + 1) % self.batches_per_update == 0:
                        self.optimizer.step()
                moving_loss += loss.item()
                try:
                    results = {key:
                                   results.get(key, 0)
                                   + val.mean(0).detach().cpu().numpy() if not torch.isnan(val.mean())
                                   else results.get(key, 0) * (iter_id + 1) / iter_id
                                   if torch.nonzero(torch.isnan(val)).nelement() == val.nelement() else
                                   results.get(key, 0) + (
                                               val[1 - torch.isnan(val)] / (1 - torch.isnan(val).float()).sum()).item()
                               for (key, val) in loss_stats.items()}
                except:
                    a = 1
                del loss, loss_stats
                bar_object.set_postfix_str("{phase}:[{epoch}]' loss={loss} {losses}".
                                           format(phase=phase, epoch=epoch,
                                                  loss=moving_loss * self.batches_per_update / (iter_id + 1),
                                                  losses={k: np.array2string(v / (iter_id + 1), precision=3)
                                                          for k, v in results.items()}))
                bar_object.update(1)
                del output
        results = {k: v / num_iters for k, v in results.items()}
        results.update({'loss': moving_loss / num_iters})
        return results

    def val(self, epoch, data_loader):
        return self.run_epoch('val', epoch, data_loader)

    def train(self, epoch, data_loader):
        return self.run_epoch('train', epoch, data_loader)

    def _predict(self, dataloader, tta=None, mapper=None, label_mapper=None, aggregator=None):
        model = DataParallel(self.model_with_loss.module.model.eval(), device_ids=[0, 1]).eval()
        for i, batch in enumerate(dataloader):
            outputs = []
            batch = self.to(batch)
            with torch.no_grad():
                inp = batch["input"] if isinstance(batch, dict) else batch[0]
                if len(inp.shape) > 4:
                    tta = inp.shape[1]
                    inp = inp.view(-1, inp.shape[-3], inp.shape[-2], inp.shape[-1])
                t_output = model(inp)  #
                if not isinstance(t_output, list):
                    t_output = [{"y": t_output}]
                if tta:  # [0]
                    t_output = [{key: val.view(
                        *tuple([val.shape[0] // tta, tta] + [el for el in val.shape[1:]])).sigmoid().mean(1)
                                 for key, val in t_output[0].items()}]
            data = mapper(t_output, 1, batch["meta"] if " meta" in batch else None) if mapper else t_output  # !!!!
            outputs.extend(data)
            aggregation = aggregator(outputs) if aggregator else outputs
            labels = label_mapper(batch) if label_mapper else None
            yield (inp.cpu(), aggregation, labels) if label_mapper else (inp.cpu(), aggregation)

    def _debug_run(self, dataloader, mapper=None, label_mapper=None, aggregator=None):
        for i, batch in enumerate(dataloader):
            outputs = []
            batch = self.to(batch)
            data = mapper(batch, 1, i) if mapper else batch  # !!!!
            outputs.extend(data)
            aggregation = aggregator(outputs) if aggregator else outputs
            labels = label_mapper(batch) if label_mapper else None
            yield (aggregation, labels) if label_mapper else (batch["input"].cpu(), aggregation)

    def _test(self, dataloader, gt_key, tta=None, mapper=None, aggregator=None, verbose=0):
        self.metric_handler.verbose = verbose
        return self.metric_handler(self._predict(dataloader,
                                                 tta=tta,
                                                 mapper=mapper,
                                                 label_mapper=lambda x: x.get(gt_key, None),
                                                 aggregator=aggregator))
