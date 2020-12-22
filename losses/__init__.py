from utils import create_instance, import_all_from_directory

import_all_from_directory(__name__)


def create_loss(loss_name, **kwargs):
    return create_instance(loss_name, globals(), **kwargs)
