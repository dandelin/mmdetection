import torch
import torch.nn as nn
from torchvision import models
import requests
import shutil
import os
import ipdb
import sys
from loguru import logger
from torch.nn.modules.batchnorm import _BatchNorm

sys.path.append(
    os.path.join(
        os.path.dirname(
            os.path.dirname(
                os.path.dirname(
                    os.path.dirname(
                        os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
                    )
                )
            )
        ),
        os.pardir,
    )
)
from classification.efficientnet import EfficientNet
from ..registry import BACKBONES


@BACKBONES.register_module
class EfficientNetB5(nn.Module):
    """ Feature extractor class for efficientnet """

    def __init__(self, min_reduction=4, frozen_stages=-1):
        super().__init__()

        self.model = EfficientNet.from_name("efficientnet-b5")
        self.min_reduction = min_reduction
        self.frozen_stages = frozen_stages
        # self.train()

    @staticmethod
    def act(x):
        """ Swish activation function """
        return x * torch.sigmoid(x)

    def forward(self, x):
        input_sz = x.shape
        # Stem
        x = self.act(self.model._bn0(self.model._conv_stem(x)))

        # Blocks
        outputs = []
        for idx, block in enumerate(self.model._blocks):
            drop_connect_rate = self.model._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self.model._blocks)
            x = block(x, drop_connect_rate)

            small_enough = x.shape[-1] <= (input_sz[-1] / self.min_reduction) + 1
            size_not_encountered = x.shape[-1] not in [_o.shape[-1] for _o in outputs]
            if small_enough:
                if size_not_encountered:
                    outputs.append(x)
                else:
                    outputs[-1] = x

        return tuple(outputs)

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str) or pretrained is None:
            try:
                rank = torch.distributed.get_rank()
            except:
                rank = 0
            weight_url = (
                f"https://twg.daumcdn.net/ojo_tf_model_zoo/OX/efficientnet-b5.pth"
            )
            weight_path = f"result/model_pretrained/efficientnet-b5_{rank}.pth"
            os.makedirs("result/model_pretrained", exist_ok=True)

            if not os.path.exists(weight_path):
                response = requests.get(weight_url, stream=True)
                logger.info(f"Downloaded pretrained weights from {weight_url}.")
                with open(weight_path, "wb") as out_file:
                    shutil.copyfileobj(response.raw, out_file)
                    logger.info(f"Saved pretrained weights to {weight_path}.")
                del response

            self.model.load_state_dict(torch.load(weight_path))
            logger.info(f"Loaded model from {weight_path}")
        else:
            raise TypeError("pretrained must be a str or None")

    # def train(self, mode=True):
    #     super().train(mode)
    #     if mode:
    #         for m in self.modules():
    #             # trick: eval have effect on BatchNorm only
    #             if isinstance(m, _BatchNorm):
    #                 m.eval()
    #             # shuts down all parameters, training only neck and head
    #             # for param in m.parameters():
    #             #     param.requires_grad = False
    #
    # def _freeze_stages(self):
    #     if self.frozen_stages >= 0:
    #         self.model._bn0.eval()
    #         for m in [self.model._conv_stem, self.model._bn0]:
    #             for param in m.parameters():
    #                 param.requires_grad = False
    #
    #     # only apply frozen_stages = 1, blocks are hard-coded
    #     # NO, just try all-finetunable network
    #
    #     for i, m in enumerate(self.model._blocks):
    #         pass
