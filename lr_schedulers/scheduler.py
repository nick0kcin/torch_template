from torch.optim.lr_scheduler import *
from utils import create_instance, get_class


def create_lr_schedule(name, optimizer, **kwargs):
    return create_instance(name, globals(), optimizer=optimizer, **kwargs)


def get_lr_schedule_class(name):
    return get_class(name, globals())
