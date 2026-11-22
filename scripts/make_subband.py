import json
import os

import numpy as np
from PIL import Image


def haar2(a):
    H, W = a.shape[:2]
    H2, W2 = H // 2, W // 2
    a = a[:H2 * 2, :W2 * 2]
    x00 = a[0::2, 0::2]; x01 = a[0::2, 1::2]
    x10 = a[1::2, 0::2]; x11 = a[1::2, 1::2]
    LL = (x00 + x01 + x10 + x11) / 2.0
    LH = (x00 + x01 - x10 - x11) / 2.0
    HL = (x00 - x01 + x10 - x11) / 2.0
    HH = (x00 - x01 - x10 + x11) / 2.0
    return LL, LH, HL, HH


def ihaar(LL, LH, HL, HH):
    H2, W2 = LL.shape[:2]
    out = np.zeros((H2 * 2, W2 * 2, LL.shape[2]), dtype=np.float32)
    out[0::2, 0::2] = (LL + LH + HL + HH) / 2.0
    out[0::2, 1::2] = (LL + LH - HL - HH) / 2.0
    out[1::2, 0::2] = (LL - LH + HL - HH) / 2.0
    out[1::2, 1::2] = (LL - LH - HL + HH) / 2.0
    return out


targets = {"dropLH": 1, "dropHL": 2, "dropHH": 3}
val = json.load(open("/home/ubuntu/ws/data/fog_voc/annotations/val.json"))
src = "/home/ubuntu/ws/data/fog_voc/JPEGImages"

for name, idx in targets.items():
    out = f"/home/ubuntu/ws/data/{name}_voc/JPEGImages"
    os.makedirs(out, exist_ok=True)
    for it in val["images"]:
        a = np.asarray(Image.open(os.path.join(src, it["file_name"])).convert("RGB"), dtype=np.float32) / 255.0
        bands = list(haar2(a))
        bands[idx] = np.zeros_like(bands[idx])
        rec = np.clip(ihaar(*bands), 0, 1)
        Image.fromarray((rec * 255).astype(np.uint8)).save(os.path.join(out, it["file_name"]), quality=95)
    os.makedirs(f"/home/ubuntu/ws/data/{name}_voc/annotations", exist_ok=True)
    json.dump(val, open(f"/home/ubuntu/ws/data/{name}_voc/annotations/val.json", "w"))
    print(name, "done")
