# Copyright (c) Facebook, Inc. and its affiliates.
import os

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import load_sem_seg

COCO_CATEGORIES = [
    {"color": [0, 0, 0], "isthing": 1, "id": 1, "name": "unlabelled"},
    {"color": [0, 0, 255], "isthing": 1, "id": 2, "name": "Hand-Drill"},
    {"color": [0, 255, 0], "isthing": 1, "id": 3, "name": "Backpack"},
    {"color": [255, 0, 0], "isthing": 1, "id": 4, "name": "Fire-Extinguisher"},
    {"color": [255, 255, 255], "isthing": 1, "id": 5, "name": "Survivor"},
]


def _get_msrs_stuff_meta():
    # Id 0 is reserved for ignore_label, we change ignore_label for 0
    # to 255 in our pre-processing.
    stuff_ids = [k["id"] for k in COCO_CATEGORIES]
    assert len(stuff_ids) == 5, len(stuff_ids)

    # For semantic segmentation, this mapping maps from contiguous stuff id
    # (in [0, 91], used in models) to ids in the dataset (used for processing results)
    stuff_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(stuff_ids)}
    stuff_classes = [k["name"] for k in COCO_CATEGORIES]
    stuff_colors = [k["color"] for k in COCO_CATEGORIES]

    ret = {
        "stuff_dataset_id_to_contiguous_id": stuff_dataset_id_to_contiguous_id,
        "stuff_classes": stuff_classes,
        "stuff_colors": stuff_colors,
    }
    return ret


def register_all_msrs_stuff(root):
    # root = os.path.join(root, "coco", "coco_stuff_10k")
    meta = _get_msrs_stuff_meta()
    for name, image_dirname, sem_seg_dirname in [
        ("train", "train/Visible", "train/Label"),
        ("test", "test/Visible", "test/Label"),
    ]:
        image_dir = os.path.join(root, image_dirname)
        gt_dir = os.path.join(root, sem_seg_dirname)
        name = f"pst_{name}_sem_seg"
        DatasetCatalog.register(
            name, lambda x=image_dir, y=gt_dir: load_sem_seg(y, x, gt_ext="png", image_ext="jpg")
        )
        MetadataCatalog.get(name).set(
            image_root=image_dir,
            sem_seg_root=gt_dir,
            evaluator_type="sem_seg",
            ignore_label=255,
            **meta,
        )


_root = "/space0/liuchang/VI/RGBTdataset/PST900_RGBT_Dataset"
register_all_msrs_stuff(_root)