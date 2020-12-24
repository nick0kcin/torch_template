from torch.utils.data.sampler import *
from torch.utils.data.dataloader import *
from torch.utils.data.dataset import *
from catalyst.data.sampler import BalanceClassSampler
from utils import create_instance, import_all_from_directory

import_all_from_directory(__name__)


def create_dataset(name, **kwargs):
    return create_instance(name, globals(), **kwargs)


def create_dataloader(name, dataset, sampler=None,  collate=None, **kwargs):
    if collate:
        return create_instance(name, globals(), dataset=dataset, sampler=sampler, collate_fn=collate, **kwargs)
    else:
        return create_instance(name, globals(), dataset=dataset, sampler=sampler, **kwargs)


def create_sampler(name, dataset, kwargs):
    loc = locals()
    nargs = {}
    for key, expr in kwargs.items():
        try:
            nargs.update({key: eval(str(expr), globals(), loc)})
        except NameError:
            nargs.update({key: expr})
    # nargs = {key: eval(str(expr), globals(), loc) for key, expr in kwargs.items()}
    return create_instance(name, globals(), **nargs)