from torch.optim.lr_scheduler import *
from utils import create_instance


def create_lr_schedule(name, optimizer, **kwargs):
    return create_instance(name, globals(), optimizer=optimizer, **kwargs)



