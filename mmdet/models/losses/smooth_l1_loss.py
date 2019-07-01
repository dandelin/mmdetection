import torch.nn as nn
import sys
import os

sys.path.append(
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))),
        os.pardir,
    )
)
from mmdet.core import weighted_smoothl1

from ..registry import LOSSES


@LOSSES.register_module
class SmoothL1Loss(nn.Module):
    def __init__(self, beta=1.0, loss_weight=1.0):
        super(SmoothL1Loss, self).__init__()
        self.beta = beta
        self.loss_weight = loss_weight

    def forward(self, pred, target, weight, *args, **kwargs):
        loss_bbox = self.loss_weight * weighted_smoothl1(
            pred, target, weight, beta=self.beta, *args, **kwargs
        )
        return loss_bbox
