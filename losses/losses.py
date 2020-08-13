from .focal_loss import FocalLoss
from .cat_loss import  CatLoss
from .cosine_loss import CosineLoss
from .bin_angle_loss import  BinAngleLoss
from .bin_expec_angle import  BinExpectationAngleLoss
from .l1loss import L1Loss
from .squared_hinge import Hinge
from .BCE import BCE
from .l1sparseloss import L1SparseLoss
from .seg_loss import get_seg_loss
from .corellation_loss import CorellationLoss
from torch.nn.functional import binary_cross_entropy_with_logits
from .pair_loss import PairLoss
from  .LabelSmoothing import *
from utils import create_instance


def create_loss(loss_name, **kwargs):
    return create_instance(loss_name, globals(), **kwargs)
