from torch.optim.lr_scheduler import _LRScheduler


def compose_wrapper(classes, args, kwargs, times, lrs):
    """
    :param classes: list of lr scheduler classes
    :param args:  positional arguments for class instances
    :param kwargs: keyword arguments for class instances
    :param times: list of  switching epochs
    :return: Composer class
    """
    steps, classes, args, kwargs, lrs = list(zip(*sorted(zip(times, classes, args, kwargs, lrs), reverse=True)))
    steps, classes, args, kwargs, lrs = list(steps), list(classes), list(args), list(kwargs), list(lrs)
    unique_list = list(set(classes))

    for class_object in classes:
        if not issubclass(class_object, _LRScheduler):
            raise ValueError("all lr schedulers must be children of _LRScheduler")

    class Compose_(*unique_list):
        def __init__(self, optimizer):
            index = unique_list.index(classes[-1])
            # super(Compose_, self).__init__(optimizer)
            self.class_type = unique_list[index - 1] if index else Compose_
            # _LRScheduler init
            # super(unique_list[-1], self).__init__(optimizer)
            super(self.class_type, self).__init__(optimizer, *(args[-1]), **(kwargs[-1]))
            if lrs[-1]:
                for i, param_group in enumerate(optimizer.param_groups):
                    param_group['lr'] = lrs[-1][i]
                    param_group['initial_lr'] = lrs[-1][i]

        def get_lr(self):
            if len(steps) <= 1 or self.last_epoch < steps[-2]:
                return super(self.class_type, self).get_lr()
            while len(steps) > 1 and self.last_epoch >= steps[-2]:  # switch to next lr scheduler
                # if len(steps) <= 1 or self.last_epoch < steps[-2]:
                #     break
                index = unique_list.index(classes[-2])
                new_class = unique_list[index - 1] if index else Compose_
                steps.pop()
                args.pop()
                kwargs.pop()
                classes.pop()
                lrs.pop()
                self.last_epoch -= steps[-1]
                for i in range(len(steps)):
                    steps[i] -= steps[-1]
                # lr initialize
                if lrs[-1]:
                    for i, param_group in enumerate(self.optimizer.param_groups):
                        param_group['lr'] = lrs[-1][i]
                        param_group['initial_lr'] = lrs[-1][i]
                # _LRScheduler init
            # super(unique_list[-1], self).__init__(self.optimizer, last_epoch=self.last_epoch)
            # new schedule init
            super(new_class, self).__init__(self.optimizer, *(args[-1]),  **(kwargs[-1]), last_epoch=self.last_epoch)
            self.class_type = new_class
            # self.last_epoch = last_epoch
            return super(self.class_type, self).get_lr()
    return Compose_
