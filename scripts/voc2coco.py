"""VOC xml -> COCO json for the adverse-weather detection benchmark.

python voc2coco.py --root ~/ws/data/fog_voc --out ~/ws/data/fog_voc/annotations --val-ratio 0.2
"""

from __future__ import annotations

import argparse
import json
import os
import random
import xml.etree.ElementTree as ET

VOC_CLASSES = [
    "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
    "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor",
]


def convert(root: str, out_dir: str, val_ratio: float, seed: int = 0) -> None:
    ann_dir = os.path.join(root, "Annotations")
    img_dir = os.path.join(root, "JPEGImages")
    stems = sorted(f[:-4] for f in os.listdir(ann_dir) if f.endswith(".xml"))
    random.Random(seed).shuffle(stems)
    n_val = int(len(stems) * val_ratio)
    splits = {"train": stems[n_val:], "val": stems[:n_val]}

    os.makedirs(out_dir, exist_ok=True)
    cats = [{"id": i + 1, "name": n, "supercategory": "none"} for i, n in enumerate(VOC_CLASSES)]

    for split, ids in splits.items():
        images, anns, ann_id = [], [], 1
        for idx, stem in enumerate(ids):
            tree = ET.parse(os.path.join(ann_dir, stem + ".xml"))
            r = tree.getroot()
            size = r.find("size")
            w = int(size.find("width").text)
            h = int(size.find("height").text)
            images.append({"id": idx, "file_name": stem + ".jpg", "width": w, "height": h})
            for obj in r.findall("object"):
                name = obj.find("name").text
                if name not in VOC_CLASSES:
                    continue
                b = obj.find("bndbox")
                x1, y1 = float(b.find("xmin").text), float(b.find("ymin").text)
                x2, y2 = float(b.find("xmax").text), float(b.find("ymax").text)
                bw, bh = max(0.0, x2 - x1), max(0.0, y2 - y1)
                if bw < 1 or bh < 1:
                    continue
                anns.append({
                    "id": ann_id, "image_id": idx,
                    "category_id": VOC_CLASSES.index(name) + 1,
                    "bbox": [x1, y1, bw, bh], "area": bw * bh, "iscrowd": 0,
                })
                ann_id += 1
        coco = {"images": images, "annotations": anns, "categories": cats}
        path = os.path.join(out_dir, f"{split}.json")
        with open(path, "w") as f:
            json.dump(coco, f)
        print(f"{split}: {len(images)} images, {len(anns)} boxes -> {path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--val-ratio", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    convert(a.root, a.out, a.val_ratio, a.seed)
