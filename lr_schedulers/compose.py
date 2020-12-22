from torch.optim.lr_scheduler import _LRScheduler
from bisect import bisect_right


class ComposeLrSchedulers(_LRScheduler):

    def __init__(self, optimizer, schedulers, starts, lrs):
        self.optimizer = optimizer
        self.schedulers = schedulers
        for schedule, lr in zip(self.schedulers, lrs):
            schedule.base_lrs = lr
        self.starts = starts
        super(ComposeLrSchedulers, self).__init__(optimizer)
        for i, lr in enumerate(self.get_lr()):
            self.optimizer.param_groups[i]["lr"] = lr

    def step(self, epoch=None):
        new_epoch = self.last_epoch + 1 if epoch is None else epoch
        idx = bisect_right(self.starts, new_epoch)
        if epoch is None:
            if new_epoch - self.starts[idx - 1]:
                self.schedulers[idx - 1].step()
        else:
            self.schedulers[idx - 1].step(new_epoch - self.starts[idx - 1])
        self.last_epoch = new_epoch

    def get_lr(self):
        idx = bisect_right(self.starts, self.last_epoch)
        return self.schedulers[idx - 1]._get_closed_form_lr()

    @property
    def current_scheduler(self):
        idx = bisect_right(self.starts, self.last_epoch)
        return self.schedulers[idx - 1]
