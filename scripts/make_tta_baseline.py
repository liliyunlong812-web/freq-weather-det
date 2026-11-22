import json, os
import numpy as np
from PIL import Image
import cv2

val = json.load(open("/home/ubuntu/ws/data/fog_voc/annotations/val.json"))
src = "/home/ubuntu/ws/data/fog_voc/JPEGImages"
for name in ["clahe", "gamma"]:
    out = f"/home/ubuntu/ws/data/{name}_voc/JPEGImages"
    os.makedirs(out, exist_ok=True)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    for it in val["images"]:
        a = np.asarray(Image.open(os.path.join(src, it["file_name"])).convert("RGB"))
        if name == "clahe":
            lab = cv2.cvtColor(a, cv2.COLOR_RGB2LAB)
            lab[..., 0] = clahe.apply(lab[..., 0])
            b = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        else:
            b = np.clip(((a / 255.0) ** 0.7) * 255, 0, 255).astype(np.uint8)
        Image.fromarray(b).save(os.path.join(out, it["file_name"]), quality=95)
    os.makedirs(f"/home/ubuntu/ws/data/{name}_voc/annotations", exist_ok=True)
    json.dump(val, open(f"/home/ubuntu/ws/data/{name}_voc/annotations/val.json", "w"))
    print(name, "done")