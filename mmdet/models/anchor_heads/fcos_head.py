import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import normal_init
import sys
import os
import ipdb

sys.path.insert(0,
    os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
    )
)

from mmdet.core import (
    sigmoid_focal_loss,
    iou_loss,
    multi_apply,
    multiclass_nms,
    distance2bbox,
)
from ..registry import HEADS
from ..utils import bias_init_with_prob, Scale, ConvModule

INF = 1e8


@HEADS.register_module
class FCOSHead(nn.Module):
    def __init__(
        self,
        num_classes,
        in_channels,
        feat_channels=256,
        stacked_convs=4,
        strides=(4, 8, 16, 32, 64),
        regress_ranges=((-1, 64), (64, 128), (128, 256), (256, 512), (512, INF)),
        conv_cfg=None,
        norm_cfg=dict(type="GN", num_groups=32, requires_grad=True),
        activation="relu",
        double_head=None,
    ):
        super(FCOSHead, self).__init__()

        self.num_classes = num_classes
        self.cls_out_channels = num_classes - 1
        self.in_channels = in_channels
        self.feat_channels = feat_channels
        self.stacked_convs = stacked_convs
        self.strides = strides
        self.regress_ranges = regress_ranges
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.activation = activation
        self.double_head = double_head

        self._init_layers()

    def _init_layers(self):
        self.cls_convs = nn.ModuleList()
        self.attr_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            self.cls_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    bias=self.norm_cfg is None,
                    activation=self.activation,
                )
            )
            self.attr_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    bias=self.norm_cfg is None,
                    activation=self.activation,
                )
            )
            self.reg_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    bias=self.norm_cfg is None,
                    activation=self.activation,
                )
            )
        self.fcos_cls = nn.Conv2d(
            self.feat_channels, self.cls_out_channels, 3, padding=1
        )
        self.fcos_reg = nn.Conv2d(self.feat_channels, 4, 3, padding=1)
        self.fcos_attr = nn.Conv2d(self.feat_channels, 400, 3, padding=1)
        self.fcos_centerness = nn.Conv2d(self.feat_channels, 1, 3, padding=1)

        self.scales = nn.ModuleList([Scale(1.0) for _ in self.strides])

    def init_weights(self):
        for m in self.cls_convs:
            normal_init(m.conv, std=0.01)
        for m in self.attr_convs:
            normal_init(m.conv, std=0.01)
        for m in self.reg_convs:
            normal_init(m.conv, std=0.01)
        bias_cls = bias_init_with_prob(0.01)
        bias_attr = bias_init_with_prob(0.01)
        normal_init(self.fcos_cls, std=0.01, bias=bias_cls)
        normal_init(self.fcos_attr, std=0.01, bias=bias_attr)
        normal_init(self.fcos_reg, std=0.01)
        normal_init(self.fcos_centerness, std=0.01)

    def forward(self, feats, features=False):
        return multi_apply(self.forward_single, feats, self.scales, features=features)

    def forward_single(self, x, scale, features=False):
        orig_feat = x
        cls_feat = x
        attr_feat = x
        reg_feat = x

        for i, cls_layer in enumerate(self.cls_convs):
            if i == 3:
                orig_feat = cls_feat
            cls_feat = cls_layer(cls_feat)
        cls_score = self.fcos_cls(cls_feat)

        for attr_layer in self.attr_convs:
            attr_feat = attr_layer(attr_feat)
        attr_score = self.fcos_attr(attr_feat)

        for reg_layer in self.reg_convs:
            reg_feat = reg_layer(reg_feat)
        # scale the bbox_pred of different level
        bbox_pred = scale(self.fcos_reg(reg_feat)).exp()
        centerness = self.fcos_centerness(
            reg_feat if self.double_head is not None else cls_feat
        )

        if features:
            return (cls_score, bbox_pred, centerness, attr_score, orig_feat)
        else:
            return cls_score, bbox_pred, centerness, attr_score

    def loss(
        self,
        cls_scores,
        bbox_preds,
        centernesses,
        attr_scores,
        gt_bboxes,
        gt_labels,
        gt_attributes,
        img_metas,
        cfg,
        gt_bboxes_ignore=None,
    ):
        assert (
            len(cls_scores) == len(bbox_preds) == len(centernesses) == len(attr_scores)
        )
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        all_level_points = self.get_points(
            featmap_sizes, bbox_preds[0].dtype, bbox_preds[0].device
        )
        labels, bbox_targets, attrs = self.fcos_target(
            all_level_points, gt_bboxes, gt_labels, gt_attributes
        )

        num_imgs = cls_scores[0].size(0)
        # flatten cls_scores, bbox_preds and centerness
        flatten_cls_scores = [
            cls_score.permute(0, 2, 3, 1).reshape(-1, self.cls_out_channels)
            for cls_score in cls_scores
        ]
        flatten_bbox_preds = [
            bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4) for bbox_pred in bbox_preds
        ]
        flatten_centerness = [
            centerness.permute(0, 2, 3, 1).reshape(-1) for centerness in centernesses
        ]
        flatten_attr_scores = [
            attr_score.permute(0, 2, 3, 1).reshape(-1, 400)
            for attr_score in attr_scores
        ]
        flatten_cls_scores = torch.cat(flatten_cls_scores)
        flatten_bbox_preds = torch.cat(flatten_bbox_preds)
        flatten_centerness = torch.cat(flatten_centerness)
        flatten_attr_scores = torch.cat(flatten_attr_scores)
        flatten_labels = torch.cat(labels)
        flatten_attrs = torch.cat(attrs)
        flatten_bbox_targets = torch.cat(bbox_targets)
        # repeat points to align with bbox_preds
        flatten_points = torch.cat(
            [points.repeat(num_imgs, 1) for points in all_level_points]
        )

        pos_inds = flatten_labels.nonzero().reshape(-1)
        num_pos = len(pos_inds)
        loss_cls = sigmoid_focal_loss(
            flatten_cls_scores, flatten_labels, cfg.gamma, cfg.alpha, "none"
        ).sum()[None] / (
            num_pos + num_imgs
        )  # avoid num_pos is 0

        pos_bbox_preds = flatten_bbox_preds[pos_inds]
        pos_bbox_targets = flatten_bbox_targets[pos_inds]
        pos_centerness = flatten_centerness[pos_inds]
        pos_attr_pred = flatten_attr_scores[pos_inds]
        pos_attr_targets = flatten_attrs[pos_inds]

        if num_pos > 0:
            pos_centerness_targets = self.centerness_target(pos_bbox_targets)
            pos_points = flatten_points[pos_inds]
            pos_decoded_bbox_preds = distance2bbox(pos_points, pos_bbox_preds)
            pos_decoded_target_preds = distance2bbox(pos_points, pos_bbox_targets)
            # centerness weighted iou loss
            loss_reg = (
                (
                    iou_loss(
                        pos_decoded_bbox_preds,
                        pos_decoded_target_preds,
                        reduction="none",
                    )
                    * pos_centerness_targets
                ).sum()
                / pos_centerness_targets.sum()
            )[None]
            loss_centerness = F.binary_cross_entropy_with_logits(
                pos_centerness, pos_centerness_targets, reduction="mean"
            )[None]
            # train those have at least one attribute
            valid_attr_idx = pos_attr_targets.sum(dim=1) != 0
            loss_attr = F.binary_cross_entropy_with_logits(
                pos_attr_pred[valid_attr_idx],
                pos_attr_targets[valid_attr_idx],
                reduction="mean",
            )[None]
            if torch.isnan(loss_attr):
                loss_attr = (
                    F.binary_cross_entropy_with_logits(
                        pos_attr_pred, pos_attr_targets, reduction="mean"
                    )[None]
                    * 0
                )
        else:
            loss_reg = flatten_bbox_preds.sum()[None] * 0
            loss_centerness = flatten_centerness.sum()[None] * 0
            loss_attr = flatten_attr_scores.sum()[None] * 0

        return dict(
            loss_cls=loss_cls,
            loss_reg=loss_reg,
            loss_centerness=loss_centerness,
            loss_attr=loss_attr,
        )

    def get_bboxes(
        self,
        cls_scores,
        bbox_preds,
        centernesses,
        attr_scores,
        img_metas,
        cfg,
        rescale=None,
        feats=None,
    ):
        assert len(cls_scores) == len(bbox_preds) == len(attr_scores)
        num_levels = len(cls_scores)

        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        mlvl_points = self.get_points(
            featmap_sizes, bbox_preds[0].dtype, bbox_preds[0].device
        )
        result_list = []
        for img_id in range(len(img_metas)):
            cls_score_list = [cls_scores[i][img_id].detach() for i in range(num_levels)]
            attr_score_list = [
                attr_scores[i][img_id].detach() for i in range(num_levels)
            ]
            bbox_pred_list = [bbox_preds[i][img_id].detach() for i in range(num_levels)]
            centerness_pred_list = [
                centernesses[i][img_id].detach() for i in range(num_levels)
            ]
            if feats is not None:
                # feats_list = [
                #     {
                #         "cls": feats[i]["cls"][img_id].detach(),
                #         "reg": feats[i]["reg"][img_id].detach(),
                #         "attr": feats[i]["attr"][img_id].detach(),
                #     }
                #     for i in range(num_levels)
                # ]
                feats_list = [feats[i][img_id].detach() for i in range(num_levels)]
            else:
                feats_list = None

            img_shape = img_metas[img_id]["img_shape"]
            scale_factor = img_metas[img_id]["scale_factor"]
            det_bboxes = self.get_bboxes_single(
                cls_score_list,
                bbox_pred_list,
                centerness_pred_list,
                attr_score_list,
                mlvl_points,
                img_shape,
                scale_factor,
                cfg,
                rescale,
                feats_list=feats_list,
            )
            result_list.append(det_bboxes)
        return result_list

    def get_bboxes_single(
        self,
        cls_scores,
        bbox_preds,
        centernesses,
        attr_scores,
        mlvl_points,
        img_shape,
        scale_factor,
        cfg,
        rescale=False,
        feats_list=None,
    ):
        assert (
            len(cls_scores) == len(bbox_preds) == len(mlvl_points) == len(attr_scores)
        )
        mlvl_bboxes = []
        mlvl_scores = []
        mlvl_centerness = []
        mlvl_attrs = []
        mlvl_feats = []
        for ii, (cls_score, bbox_pred, centerness, attr_score, points) in enumerate(
            zip(cls_scores, bbox_preds, centernesses, attr_scores, mlvl_points)
        ):
            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]
            scores = (
                cls_score.permute(1, 2, 0).reshape(-1, self.cls_out_channels).sigmoid()
            )
            centerness = centerness.permute(1, 2, 0).reshape(-1).sigmoid()
            attrs = attr_score.permute(1, 2, 0).reshape(-1, 400).sigmoid()
            if feats_list is not None:
                feats_list_single_level = feats_list[ii]
                # feats_list_single_level = {
                #     "cls": feats_list_single_level["cls"]
                #     .permute(1, 2, 0)
                #     .reshape(-1, 256),
                #     "reg": feats_list_single_level["reg"]
                #     .permute(1, 2, 0)
                #     .reshape(-1, 256),
                #     "attr": feats_list_single_level["attr"]
                #     .permute(1, 2, 0)
                #     .reshape(-1, 256),
                # }
                feats_list_single_level = feats_list_single_level.permute(
                    1, 2, 0
                ).reshape(-1, 256)
            else:
                feats_list_single_level = None

            bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 4)
            nms_pre = cfg.get("nms_pre", -1)
            if nms_pre > 0 and scores.shape[0] > nms_pre:
                max_scores, _ = (scores * centerness[:, None]).max(dim=1)
                _, topk_inds = max_scores.topk(nms_pre)
                points = points[topk_inds, :]
                bbox_pred = bbox_pred[topk_inds, :]
                scores = scores[topk_inds, :]
                centerness = centerness[topk_inds]
                attrs = attrs[topk_inds, :]
                if feats_list_single_level is not None:
                    # feats_list_single_level = {
                    #     "cls": feats_list_single_level["cls"][topk_inds, :],
                    #     "reg": feats_list_single_level["reg"][topk_inds, :],
                    #     "attr": feats_list_single_level["attr"][topk_inds, :],
                    # }
                    feats_list_single_level = feats_list_single_level[topk_inds, :]
            bboxes = distance2bbox(points, bbox_pred, max_shape=img_shape)
            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)
            mlvl_centerness.append(centerness)
            mlvl_attrs.append(attrs)
            mlvl_feats.append(feats_list_single_level)

        mlvl_bboxes = torch.cat(mlvl_bboxes)
        if rescale:
            mlvl_bboxes /= mlvl_bboxes.new_tensor(scale_factor)
        mlvl_scores = torch.cat(mlvl_scores)
        padding = mlvl_scores.new_zeros(mlvl_scores.shape[0], 1)
        mlvl_scores = torch.cat([padding, mlvl_scores], dim=1)
        mlvl_centerness = torch.cat(mlvl_centerness)
        mlvl_attrs = torch.cat(mlvl_attrs)

        if mlvl_feats[0] is not None:
            # mlvl_feats = {
            #     "cls": torch.cat([mlvl_feat["cls"] for mlvl_feat in mlvl_feats]),
            #     "reg": torch.cat([mlvl_feat["reg"] for mlvl_feat in mlvl_feats]),
            #     "attr": torch.cat([mlvl_feat["attr"] for mlvl_feat in mlvl_feats]),
            # }
            mlvl_feats = torch.cat(mlvl_feats)
        else:
            mlvl_feats = None

        res = multiclass_nms(
            mlvl_bboxes,
            mlvl_scores,
            cfg.score_thr,
            cfg.nms,
            cfg.max_per_img,
            score_factors=mlvl_centerness,
            multi_attrs=mlvl_attrs,
            multi_feats=mlvl_feats,
        )

        if mlvl_feats is None:
            det_bboxes, det_labels, det_attrs = res
            det_feats = None
        else:
            det_bboxes, det_labels, det_attrs, det_feats = res

        return det_bboxes, det_labels, det_attrs, det_feats

    def get_points(self, featmap_sizes, dtype, device):
        """Get points according to feature map sizes.

        Args:
            featmap_sizes (list[tuple]): Multi-level feature map sizes.
            dtype (torch.dtype): Type of points.
            device (torch.device): Device of points.

        Returns:
            tuple: points of each image.
        """
        mlvl_points = []
        for i in range(len(featmap_sizes)):
            mlvl_points.append(
                self.get_points_single(featmap_sizes[i], self.strides[i], dtype, device)
            )
        return mlvl_points

    def get_points_single(self, featmap_size, stride, dtype, device):
        h, w = featmap_size
        x_range = torch.arange(0, w * stride, stride, dtype=dtype, device=device)
        y_range = torch.arange(0, h * stride, stride, dtype=dtype, device=device)
        y, x = torch.meshgrid(y_range, x_range)
        points = torch.stack((x.reshape(-1), y.reshape(-1)), dim=-1) + stride // 2
        return points

    def fcos_target(self, points, gt_bboxes_list, gt_labels_list, gt_attrs_list):
        assert len(points) == len(self.regress_ranges)
        num_levels = len(points)
        # expand regress ranges to align with points
        expanded_regress_ranges = [
            points[i].new_tensor(self.regress_ranges[i])[None].expand_as(points[i])
            for i in range(num_levels)
        ]
        # concat all levels points and regress ranges
        concat_regress_ranges = torch.cat(expanded_regress_ranges, dim=0)
        concat_points = torch.cat(points, dim=0)
        # get labels and bbox_targets of each image
        labels_list, bbox_targets_list, attrs_list = multi_apply(
            self.fcos_target_single,
            gt_bboxes_list,
            gt_labels_list,
            gt_attrs_list,
            points=concat_points,
            regress_ranges=concat_regress_ranges,
        )

        # split to per img, per level
        num_points = [center.size(0) for center in points]
        labels_list = [labels.split(num_points, 0) for labels in labels_list]
        attrs_list = [attrs.split(num_points, 0) for attrs in attrs_list]
        bbox_targets_list = [
            bbox_targets.split(num_points, 0) for bbox_targets in bbox_targets_list
        ]

        # concat per level image
        concat_lvl_labels = []
        concat_lvl_attrs = []
        concat_lvl_bbox_targets = []
        for i in range(num_levels):
            concat_lvl_labels.append(torch.cat([labels[i] for labels in labels_list]))
            concat_lvl_attrs.append(torch.cat([attrs[i] for attrs in attrs_list]))
            concat_lvl_bbox_targets.append(
                torch.cat([bbox_targets[i] for bbox_targets in bbox_targets_list])
            )
        return concat_lvl_labels, concat_lvl_bbox_targets, concat_lvl_attrs

    def fcos_target_single(
        self, gt_bboxes, gt_labels, gt_attrs, points, regress_ranges
    ):
        num_points = points.size(0)
        num_gts = gt_labels.size(0)

        areas = (gt_bboxes[:, 2] - gt_bboxes[:, 0] + 1) * (
            gt_bboxes[:, 3] - gt_bboxes[:, 1] + 1
        )
        # TODO: figure out why these two are different
        # areas = areas[None].expand(num_points, num_gts)
        areas = areas[None].repeat(num_points, 1)
        regress_ranges = regress_ranges[:, None, :].expand(num_points, num_gts, 2)
        gt_bboxes = gt_bboxes[None].expand(num_points, num_gts, 4)
        xs, ys = points[:, 0], points[:, 1]
        xs = xs[:, None].expand(num_points, num_gts)
        ys = ys[:, None].expand(num_points, num_gts)

        left = xs - gt_bboxes[..., 0]
        right = gt_bboxes[..., 2] - xs
        top = ys - gt_bboxes[..., 1]
        bottom = gt_bboxes[..., 3] - ys
        bbox_targets = torch.stack((left, top, right, bottom), -1)

        # condition1: inside a gt bbox
        inside_gt_bbox_mask = bbox_targets.min(-1)[0] > 0

        # condition2: limit the regression range for each location
        max_regress_distance = bbox_targets.max(-1)[0]
        inside_regress_range = (max_regress_distance >= regress_ranges[..., 0]) & (
            max_regress_distance <= regress_ranges[..., 1]
        )

        # if there are still more than one objects for a location,
        # we choose the one with minimal area
        areas[inside_gt_bbox_mask == 0] = INF
        areas[inside_regress_range == 0] = INF
        min_area, min_area_inds = areas.min(dim=1)

        labels = gt_labels[min_area_inds]
        attrs = gt_attrs[min_area_inds]
        labels[min_area == INF] = 0
        attrs[min_area == INF] = 0
        bbox_targets = bbox_targets[range(num_points), min_area_inds]

        return labels, bbox_targets, attrs

    def centerness_target(self, pos_bbox_targets):
        # only calculate pos centerness targets, otherwise there may be nan
        left_right = pos_bbox_targets[:, [0, 2]]
        top_bottom = pos_bbox_targets[:, [1, 3]]
        centerness_targets = (left_right.min(dim=-1)[0] / left_right.max(dim=-1)[0]) * (
            top_bottom.min(dim=-1)[0] / top_bottom.max(dim=-1)[0]
        )
        return torch.sqrt(centerness_targets)
