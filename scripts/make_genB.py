"""Generator B: a deliberately different fog model from the training generator.

Differences from generator A (used in training):
  * depth proxy from inverse gradient magnitude (structure-aware) instead of a vertical ramp
  * scattering coefficient log-uniform in [2, 8] instead of uniform [1.2, 3.2]
  * coloured airlight (slightly cool) in [0.75, 0.95] instead of neutral [0.86, 1.0]
  * a small Gaussian PSF applied before scattering, emulating the point spread of fog
"""

import json
import os

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, sobel


def depth_from_structure(gray, rng):
    gx = sobel(gray, axis=1)
    gy = sobel(gray, axis=0)
    grad = np.hypot(gx, gy)
    grad = gaussian_filter(grad, sigma=5.0)
    d = 1.0 - (grad - grad.min()) / (np.ptp(grad) + 1e-6)   # smooth regions are far
    noise = gaussian_filter(rng.random(gray.shape).astype(np.float32), sigma=gray.shape[0] / 9.0)
    noise = (noise - noise.min()) / (np.ptp(noise) + 1e-6)
    return np.clip(0.7 * d + 0.3 * noise, 0.0, 1.0)


val = json.load(open("/home/ubuntu/ws/data/fog_voc/annotations/val.json"))
src = "/home/ubuntu/ws/data/VOCdevkit/VOC2012/JPEGImages"
out = "/home/ubuntu/ws/data/genB_voc/JPEGImages"
os.makedirs(out, exist_ok=True)
rng = np.random.default_rng(2026)

for it in val["images"]:
    a = np.asarray(Image.open(os.path.join(src, it["file_name"])).convert("RGB"), dtype=np.float32) / 255.0
    a = np.stack([gaussian_filter(a[..., c], 0.6) for c in range(3)], axis=-1)  # PSF
    gray = a.mean(axis=2)
    d = depth_from_structure(gray, rng)
    beta = float(np.exp(rng.uniform(np.log(2.0), np.log(8.0))))
    t = np.exp(-beta * d)[..., None]
    airlight = np.array([0.88, 0.92, 0.96], dtype=np.float32) * rng.uniform(0.8, 1.0)
    rec = a * t + airlight * (1.0 - t)
    Image.fromarray((np.clip(rec, 0, 1) * 255).astype(np.uint8)).save(
        os.path.join(out, it["file_name"]), quality=95)

os.makedirs("/home/ubuntu/ws/data/genB_voc/annotations", exist_ok=True)
json.dump(val, open("/home/ubuntu/ws/data/genB_voc/annotations/val.json", "w"))
print("genB done", len(val["images"]))
