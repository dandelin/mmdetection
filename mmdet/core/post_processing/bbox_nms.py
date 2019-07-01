import torch
import sys
import os

sys.path.append(
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))),
        os.pardir,
    )
)

from mmdet.ops.nms import nms_wrapper


def multiclass_nms(
    multi_bboxes,
    multi_scores,
    score_thr,
    nms_cfg,
    max_num=-1,
    score_factors=None,
    multi_attrs=None,
):
    """NMS for multi-class bboxes.

    Args:
        multi_bboxes (Tensor): shape (n, #class*4) or (n, 4)
        multi_scores (Tensor): shape (n, #class)
        score_thr (float): bbox threshold, bboxes with scores lower than it
            will not be considered.
        nms_thr (float): NMS IoU threshold
        max_num (int): if there are more than max_num bboxes after NMS,
            only top max_num will be kept.
        score_factors (Tensor): The factors multiplied to scores before
            applying NMS

    Returns:
        tuple: (bboxes, labels), tensors of shape (k, 5) and (k, 1). Labels
            are 0-based.
    """
    num_classes = multi_scores.shape[1]
    bboxes, labels, attrs = [], [], []
    nms_cfg_ = nms_cfg.copy()
    nms_type = nms_cfg_.pop("type", "nms")
    nms_op = getattr(nms_wrapper, nms_type)
    for i in range(1, num_classes):
        cls_inds = multi_scores[:, i] > score_thr
        if not cls_inds.any():
            continue
        # get bboxes and scores of this class
        if multi_bboxes.shape[1] == 4:
            _bboxes = multi_bboxes[cls_inds, :]
        else:
            _bboxes = multi_bboxes[cls_inds, i * 4 : (i + 1) * 4]

        _scores = multi_scores[cls_inds, i]
        if score_factors is not None:
            _scores *= score_factors[cls_inds]
        cls_dets = torch.cat([_bboxes, _scores[:, None]], dim=1)
        cls_dets, _op_ind = nms_op(cls_dets, **nms_cfg_)
        cls_labels = multi_bboxes.new_full(
            (cls_dets.shape[0],), i - 1, dtype=torch.long
        )

        if multi_attrs is not None:
            cls_attrs = multi_attrs[cls_inds, :]
            cls_attrs = cls_attrs[_op_ind]
            attrs.append(cls_attrs)

        bboxes.append(cls_dets)
        labels.append(cls_labels)

    if bboxes:
        bboxes = torch.cat(bboxes)
        labels = torch.cat(labels)

        if multi_attrs is not None:
            attrs = torch.cat(attrs)

        if bboxes.shape[0] > max_num:
            _, inds = bboxes[:, -1].sort(descending=True)
            inds = inds[:max_num]
            bboxes = bboxes[inds]
            labels = labels[inds]
            if multi_attrs is not None:
                attrs = attrs[inds]
    else:
        bboxes = multi_bboxes.new_zeros((0, 5))
        labels = multi_bboxes.new_zeros((0,), dtype=torch.long)
        if multi_attrs is not None:
            attrs = multi_bboxes.new_zeros((0, 400))

    if multi_attrs is not None:
        return bboxes, labels, attrs
    else:
        return bboxes, labels
