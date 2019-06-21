import numpy as np
import ipdb
import tqdm
import copy
import json

from sqlitedict import SqliteMultithread, SqliteDict
from collections import Counter
from collections import defaultdict
from .custom import CustomDataset


def simpson_di(data):
    N = len(data)
    data = Counter(data).values()
    return sum((n / N) ** 2 for n in data)


class UnifiedDataset(CustomDataset):
    def load_annotations(self, ann_file):
        self.split = self.img_prefix

        annotations, image_infos, simpsons_with_prefix = dict(), dict(), dict()
        for prefix in ["OPEN_IMAGES", "COCO", "PASCAL", "VISUAL_GENOME"]:
            annotations_sqlite = SqliteDict(f"{ann_file}/{prefix}/annotations.sqlite")
            image_infos_sqlite = SqliteDict(f"{ann_file}/{prefix}/image_infos.sqlite")
            annos, infos, simpsons = dict(), dict(), dict()
            for i, anno in enumerate(tqdm.tqdm(annotations_sqlite.items())):
                annos[anno[0]] = anno[1]
                labels = [box["entity"] for box in anno[1].values()]
                simpson = simpson_di(labels)
                simpsons[anno[0]] = simpson

            annotations[prefix] = annos

            for i, info in enumerate(tqdm.tqdm(image_infos_sqlite.items())):
                if info[0] in annotations[prefix]:
                    infos[info[0]] = info[1]

            image_infos[prefix] = infos
            simpsons_with_prefix[prefix] = simpsons

        entities, attributes, nats = list(), list(), list()
        for prefix, annotation in annotations.items():
            for image_id, annos in tqdm.tqdm(annotation.items()):
                for oid, entity_anno in annos.items():
                    entities.append(entity_anno["entity"])
                    nats.append(entity_anno["natural_language"])
                    attributes += entity_anno["attributes"]

        entity_counter, attribute_counter, nat_counter = (
            Counter(entities),
            Counter(attributes),
            Counter(nats),
        )
        entities = entity_counter.most_common(
            1991
        )  # at least 100 examples, total #box : 13715443
        attributes = attribute_counter.most_common(
            306
        )  # at least 500 examples, total #attribute : 12023600

        self.CLASSES = list([k for k, v in entities])
        self.ATTRIBUTES = list([k for k, v in attributes])

        self.cat_ids = self.CLASSES
        self.attr_ids = self.ATTRIBUTES
        self.cat2label = {cat_id: i + 1 for i, cat_id in enumerate(self.cat_ids)}
        self.attr2label = {attr_id: i for i, attr_id in enumerate(self.attr_ids)}

        image_ids = list()
        for prefix, annotation in annotations.items():
            for key in annotation:
                image_ids.append((prefix, key))

        self.img_ids = [i[1] for i in image_ids]

        img_infos, self.annotations, self.simpsons = list(), dict(), dict()
        for prefix, i in tqdm.tqdm(image_ids):
            info = image_infos[prefix][i]
            info["id"] = f"{prefix}__{i}"
            if prefix == "OPEN_IMAGES":
                info[
                    "filename"
                ] = f"OPEN_IMAGES/{info['split']}/{info['split']}/{i[:3]}/{i}.jpg"
            elif prefix == "PASCAL":
                info["filename"] = f"PASCAL/DATA/VOCdevkit/VOC2012/JPEGImages/{i}.jpg"
            elif prefix == "COCO":
                split = info["split"]
                info["filename"] = f"COCO/IMAGES_{split.upper()[:-4]}/{split}/{i}.jpg"
            elif prefix == "VISUAL_GENOME":
                info["filename"] = f"VISUAL_GENOME/IMAGE/{i}.jpg"
            else:
                raise NotImplementedError

            if self.split == "train":
                if "OPEN_IMAGES" in info["id"]:
                    if "train" not in info["filename"]:
                        continue

            if self.split == "val":
                if "OPEN_IMAGES" in info["id"]:
                    if "train" in info["filename"]:
                        continue
                else:
                    continue

            # filter with simpsons
            if simpsons_with_prefix[prefix][i] == 1:
                continue

            img_infos.append(info)
            self.annotations[info["id"]] = annotations[prefix][i]
            self.simpsons[info["id"]] = simpsons_with_prefix[prefix][i]

        self.img_prefix = f"{ann_file}/"

        return img_infos

    def get_ann_info(self, idx):
        gt_bboxes, gt_labels, gt_attributes = list(), list(), list()
        img_id = self.img_infos[idx]["id"]

        for i, (oid, ann) in enumerate(self.annotations[img_id].items()):
            if ann["entity"] not in self.cat_ids:
                continue
            x1, y1, w, h = (
                ann["top_left_x"],
                ann["top_left_y"],
                ann["width"],
                ann["height"],
            )
            if w * h <= 0 or w < 1 or h < 1:
                continue
            bbox = [x1, y1, x1 + w - 1, y1 + h - 1]
            gt_bboxes.append(bbox)
            gt_labels.append(self.cat2label[ann["entity"]])
            gt_attributes.append(
                [
                    self.attr2label[attr]
                    for attr in ann["attributes"]
                    if attr in self.attr2label
                    and attr
                    not in [
                        "Q690857",
                        "Q1171248",
                    ]  # ignore 'Occlusion' and 'Truncation'
                ]
            )

        if gt_bboxes:
            gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
            gt_labels = np.array(gt_labels, dtype=np.int64)
            _gt_attributes = np.zeros(
                (gt_bboxes.shape[0], len(self.ATTRIBUTES)), dtype=np.float32
            )
            for i, gt_attribute in enumerate(gt_attributes):
                _gt_attributes[i, gt_attribute] = 1
            gt_attributes = _gt_attributes
        else:
            gt_bboxes = np.zeros((0, 4), dtype=np.float32)
            gt_labels = np.array([], dtype=np.int64)
            gt_attributes = np.zeros((0, len(self.ATTRIBUTES)), dtype=np.float32)

        ann = dict(bboxes=gt_bboxes, labels=gt_labels, attributes=gt_attributes)

        return ann
