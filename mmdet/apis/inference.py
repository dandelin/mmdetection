import warnings
import os
import ipdb
from skimage.transform import resize

import mmcv
import numpy as np
import torch
from mmcv.runner import load_checkpoint
from matplotlib import pyplot as plt
import sys
import os

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))
from mmdet.core import get_classes
from mmdet.datasets import to_tensor
from mmdet.datasets.transforms import ImageTransform
from mmdet.models import build_detector


def init_detector(config, checkpoint=None, device="cuda:0"):
    """Initialize a detector from config file.

    Args:
        config (str or :obj:`mmcv.Config`): Config file path or the config
            object.
        checkpoint (str, optional): Checkpoint path. If left as None, the model
            will not load any weights.

    Returns:
        nn.Module: The constructed detector.
    """
    if isinstance(config, str):
        config = mmcv.Config.fromfile(config)
    elif not isinstance(config, mmcv.Config):
        raise TypeError(
            "config must be a filename or Config object, "
            "but got {}".format(type(config))
        )
    config.model.pretrained = None
    model = build_detector(config.model, test_cfg=config.test_cfg)
    if checkpoint is not None:
        checkpoint = load_checkpoint(model, checkpoint)
        if "CLASSES" in checkpoint["meta"]:
            model.CLASSES = checkpoint["meta"]["CLASSES"]
        else:
            warnings.warn(
                "Class names are not saved in the checkpoint's "
                "meta data, use COCO classes by default."
            )
            model.CLASSES = get_classes("coco")
    model.cfg = config  # save the config in the model for convenience
    model.to(device)
    model.eval()
    return model


def inference_detector(model, imgs, features=False):
    """Inference image(s) with the detector.

    Args:
        model (nn.Module): The loaded detector.
        imgs (str/ndarray or list[str/ndarray]): Either image files or loaded
            images.

    Returns:
        If imgs is a str, a generator will be returned, otherwise return the
        detection results directly.
    """
    cfg = model.cfg
    img_transform = ImageTransform(
        size_divisor=cfg.data.test.size_divisor, **cfg.img_norm_cfg
    )

    device = next(model.parameters()).device  # model device
    if not isinstance(imgs, list):
        return _inference_single(model, imgs, img_transform, device, features=features)
    else:
        return _inference_generator(model, imgs, img_transform, device)


def _prepare_data(img, img_transform, cfg, device):
    ori_shape = img.shape
    img, img_shape, pad_shape, scale_factor = img_transform(
        img,
        scale=cfg.data.test.img_scale,
        keep_ratio=cfg.data.test.get("resize_keep_ratio", True),
    )
    img = to_tensor(img).to(device).unsqueeze(0)
    img_meta = [
        dict(
            ori_shape=ori_shape,
            img_shape=img_shape,
            pad_shape=pad_shape,
            scale_factor=scale_factor,
            flip=False,
        )
    ]
    return dict(img=[img], img_meta=[img_meta])


def _inference_single(model, img, img_transform, device, features=False):
    img = mmcv.imread(img)
    data = _prepare_data(img, img_transform, model.cfg, device)
    with torch.no_grad():
        result = model(return_loss=False, rescale=True, features=features, **data)
    if not features:
        return result
    else:
        return result, data


def _inference_generator(model, imgs, img_transform, device):
    for img in imgs:
        yield _inference_single(model, img, img_transform, device)


# TODO: merge this method with the one in BaseDetector
def show_result(
    img_path,
    result,
    class_names,
    attr_names,
    score_thr=0.3,
    out_file=None,
    detailed=False,
):
    """Visualize the detection results on the image.

    Args:
        img (str or np.ndarray): Image filename or loaded image.
        result (tuple[list] or list): The detection result, can be either
            (bbox, segm) or just bbox.
        class_names (list[str] or tuple[str]): A list of class names.
        score_thr (float): The threshold to visualize the bboxes and masks.
        out_file (str, optional): If specified, the visualization result will
            be written to the out file instead of shown in a window.
    """
    assert isinstance(class_names, (tuple, list))
    img = mmcv.imread(img_path)
    bbox_result, attr_result = result
    bboxes = np.vstack(bbox_result)
    attrs = np.vstack(attr_result)

    # draw bounding boxes
    labels = [
        np.full(bbox.shape[0], i, dtype=np.int32) for i, bbox in enumerate(attr_result)
    ]
    labels = np.concatenate(labels)

    detailed_visualization(
        img.copy(),
        bboxes,
        labels,
        attrs,
        class_names=class_names,
        attr_names=attr_names,
        score_thr=score_thr,
        show=out_file is None,
        out_file=out_file,
        img_path=img_path,
    )
    visualize(
        img.copy(),
        bboxes,
        labels,
        attrs,
        class_names=class_names,
        attr_names=attr_names,
        score_thr=score_thr,
        show=out_file is None,
        out_file=out_file,
        img_path=img_path,
    )


def visualize(
    img,
    bboxes,
    labels,
    attrs,
    class_names=None,
    attr_names=None,
    score_thr=0,
    bbox_color="green",
    text_color="green",
    thickness=1,
    font_scale=0.5,
    show=True,
    win_name="",
    wait_time=0,
    out_file=None,
    img_path="",
):

    fig = plt.figure(dpi=300)
    ax = fig.add_subplot(1, 1, 1)
    ax.imshow(img[:, :, [2, 1, 0]])

    if score_thr > 0:
        assert bboxes.shape[1] == 5
        scores = bboxes[:, -1]
        inds = scores > score_thr
        bboxes = bboxes[inds, :]
        labels = labels[inds]
        attrs = attrs[inds, :]
        scores = scores[inds]

    for bbox, label, attr, score in zip(bboxes, labels, attrs, scores):
        bbox_int = bbox.astype(np.int32)
        x, y, w, h = (
            bbox_int[0],
            bbox_int[1],
            bbox_int[2] - bbox_int[0],
            bbox_int[3] - bbox_int[1],
        )

        ax.add_patch(
            plt.Rectangle(
                (x, y), w, h, facecolor="none", edgecolor="red", linewidth=0.5
            )
        )
        desc = f'[{score:.2f} {class_names[label]}] ({" ".join([attr_names[i] for i, sc in enumerate(attr) if sc > 0.5])})'

        bbox_style = {"facecolor": "white", "alpha": 0.5, "pad": 0}
        ax.text(x, y, desc, style="italic", bbox=bbox_style, fontsize=4)

    plt.autoscale()
    if out_file is None:
        os.makedirs(f"visualizations", exist_ok=True)
        plt.savefig(
            f"visualizations/{img_path.split('/')[-1]}_bbox_{len(bboxes)}.full.png",
            dpi=720,
        )
    else:
        plt.savefig(f"{out_file}.full.png", dpi=720)
    plt.close(fig)


def detailed_visualization(
    img,
    bboxes,
    labels,
    attrs,
    class_names=None,
    attr_names=None,
    score_thr=0,
    bbox_color="green",
    text_color="green",
    thickness=1,
    font_scale=0.5,
    show=True,
    win_name="",
    wait_time=0,
    out_file=None,
    img_path="",
):

    fig = plt.figure(figsize=(10, 100))

    if score_thr > 0:
        assert bboxes.shape[1] == 5
        scores = bboxes[:, -1]
        inds = scores > score_thr
        bboxes = bboxes[inds, :]
        labels = labels[inds]
        attrs = attrs[inds, :]
        scores = scores[inds]

    ax = fig.add_subplot(len(bboxes) + 1, 1, 1)
    # ax.imshow(resize(img[:, :, [2, 1, 0]], (224, 224), anti_aliasing=True))
    ax.imshow(img[:, :, [2, 1, 0]])

    for i, (bbox, label, attr, score) in enumerate(zip(bboxes, labels, attrs, scores)):
        ax = fig.add_subplot(len(bboxes) + 1, 1, i + 2)
        bbox_int = bbox.astype(np.int32)
        x, y, w, h = (
            bbox_int[0],
            bbox_int[1],
            bbox_int[2] - bbox_int[0],
            bbox_int[3] - bbox_int[1],
        )
        cropped = img[y : y + h, x : x + w, [2, 1, 0]]
        # ax.imshow(resize(cropped, (224, 224), anti_aliasing=True))
        ax.imshow(cropped)

        desc = f'[{score:.2f} {class_names[label]}] ({" ".join([attr_names[i] for i, sc in enumerate(attr) if sc > 0.5])})'
        bbox_style = {"facecolor": "white", "alpha": 0.5, "pad": 0}
        ax.text(0, 0, desc, style="italic", bbox=bbox_style, fontsize=12)

    plt.tight_layout()
    # plt.autoscale()
    if out_file is None:
        os.makedirs(f"visualizations", exist_ok=True)
        plt.savefig(
            f"visualizations/{img_path.split('/')[-1]}_bbox_{len(bboxes)}.part.png"
        )
    else:
        plt.savefig(f"{out_file}.part.png")
    plt.close(fig)
