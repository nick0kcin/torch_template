from torch.optim import *
from utils import create_instance


def create_optimizer(name, parameters, **kwargs):
    return create_instance(name, globals(), parameters, **kwargs)
