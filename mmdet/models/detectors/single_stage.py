import torch.nn as nn

from .base import BaseDetector
from .. import builder
from ..registry import DETECTORS
import sys
import os
import ipdb

sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))),
        os.pardir,
    ),
)
from mmdet.core import bbox2result


@DETECTORS.register_module
class SingleStageDetector(BaseDetector):
    def __init__(
        self,
        backbone,
        neck=None,
        bbox_head=None,
        train_cfg=None,
        test_cfg=None,
        pretrained=None,
    ):
        super(SingleStageDetector, self).__init__()
        self.backbone = builder.build_backbone(backbone)
        if neck is not None:
            self.neck = builder.build_neck(neck)
        self.bbox_head = builder.build_head(bbox_head)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.init_weights(pretrained=pretrained)

    def init_weights(self, pretrained=None):
        super(SingleStageDetector, self).init_weights(pretrained)
        self.backbone.init_weights(pretrained=pretrained)
        if self.with_neck:
            if isinstance(self.neck, nn.Sequential):
                for m in self.neck:
                    m.init_weights()
            else:
                self.neck.init_weights()
        self.bbox_head.init_weights()

    def extract_feat(self, img):
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x

    def forward_train(
        self,
        img,
        img_metas,
        gt_bboxes,
        gt_labels,
        gt_bboxes_ignore=None,
        gt_attributes=None,
    ):
        x = self.extract_feat(img)
        outs = self.bbox_head(x)
        loss_inputs = outs + (
            gt_bboxes,
            gt_labels,
            gt_attributes,
            img_metas,
            self.train_cfg,
        )
        losses = self.bbox_head.loss(*loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
        return losses

    def simple_test(self, img, img_meta, rescale=False, features=False):
        x = self.extract_feat(img)
        outs = self.bbox_head(x, features=features)
        if features:
            outs, feats = outs[:-1], outs[-1]
        else:
            outs, feats = outs, None

        bbox_inputs = outs + (img_meta, self.test_cfg, rescale, feats)
        bbox_list = self.bbox_head.get_bboxes(*bbox_inputs)
        bbox_results = [
            bbox2result(
                det_bboxes,
                det_labels,
                self.bbox_head.num_classes,
                attrs=det_attrs,
                feats=det_feats,
            )
            for det_bboxes, det_labels, det_attrs, det_feats in bbox_list
        ]
        return bbox_results[0]

    def aug_test(self, imgs, img_metas, rescale=False):
        raise NotImplementedError
