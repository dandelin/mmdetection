import torch
import torch.nn as nn
from torchvision import models
import requests
import shutil
import os
from loguru import logger
from classification.efficientnet import EfficientNet
from ..registry import BACKBONES


@BACKBONES.register_module
class EfficientNetB5(nn.Module):
    """ Feature extractor class for efficientnet """

    def __init__(self, min_reduction=4):
        super().__init__()

        self.model = EfficientNet.from_name("efficientnet-b5")

        self.min_reduction = min_reduction

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
        if isinstance(pretrained, str):
            weight_url = (
                f"https://twg.daumcdn.net/ojo_tf_model_zoo/OX/efficientnet-b5.pth"
            )
            weight_path = f"result/model_pretrained/efficientnet-b5.pth"
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
