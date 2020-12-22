import torch
from utils import create_instance, import_all_from_directory
try:
    from apex import amp
    APEX = True
except ModuleNotFoundError:
    APEX = False


import_all_from_directory(__name__)


def create_model(arch, **kwargs):
    model = create_instance(arch, globals(), **kwargs)
    return model


def load_model(model, model_path, optimizer=None, resume=False, epoch=None):
    start_epoch = 0
    checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
    print('loaded {}, epoch {}'.format(model_path, checkpoint['epoch']))
    state_dict_ = checkpoint['state_dict']
    state_dict = {}

    for k in state_dict_:
        if k.startswith('module') and not k.startswith('module_list'):
            state_dict[k[7:]] = state_dict_[k]
        else:
            state_dict[k] = state_dict_[k]
    model_state_dict = model.state_dict()

    for k in state_dict:
        if k in model_state_dict:
            if state_dict[k].shape != model_state_dict[k].shape:
                print('Skip loading parameter {}, required shape{}, ' \
                      'loaded shape{}.'.format(
                    k, model_state_dict[k].shape, state_dict[k].shape))
                state_dict[k] = model_state_dict[k]
        elif "backbone." + k in model_state_dict:
            if state_dict[k].shape != model_state_dict["backbone." + k].shape:
                state_dict[k] = model_state_dict["backbone." + k]
                print('Skip loading parameter {}, required shape{}, ' \
                      'loaded shape{}.'.format(
                    k, model_state_dict["backbone." + k].shape, state_dict[k].shape))
        else:
            print('Drop parameter {}.'.format(k))

    for k in model_state_dict:
        if ".".join(k.split(".")[1:]) in state_dict:
            state_dict[k] = state_dict[".".join(k.split(".")[1:])]
            del state_dict[".".join(k.split(".")[1:])]
        elif not (k in state_dict):
            print('No param {}.'.format(k))
            state_dict[k] = model_state_dict[k]
    model.load_state_dict(state_dict, strict=False)
    # amp.load_state_dict(checkpoint['amp'])
    if optimizer is not None and resume:
        try:
            optimizer.load_state_dict(checkpoint['optimizer'])
        except KeyError:
            print('No optimizer parameters in checkpoint.')
        except ValueError as e:
            print(e)
        start_epoch = epoch if epoch else checkpoint['epoch']
    value = float("inf") if not epoch else checkpoint['value']
    if optimizer is not None:
        return model, optimizer, start_epoch, value
    else:
        return model


def save_model(path, epoch, model, value, optimizer=None):
    if isinstance(model, torch.nn.DataParallel):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
    data = {'epoch': epoch,
            'value': value,
            'state_dict': state_dict
            # ,'amp': amp.state_dict()
            }
    if not (optimizer is None):
        data['optimizer'] = optimizer.state_dict()
    torch.save(data, path)
