"""Synthesise adverse-weather variants of VOC2012 for detection training.

fog : atmospheric scattering  I = J * t + A * (1 - t),  t = exp(-beta * d)
dark: low light               I = s * (J ** gamma)

d is a cheap depth proxy (vertical ramp + smoothed noise). Dependencies: numpy, PIL, scipy.
"""

from __future__ import annotations

import argparse
import os
import random
import shutil

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter


def depth_proxy(h, w, rng, strength=0.35):
    ramp = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None].repeat(w, axis=1)
    noise = gaussian_filter(rng.random((h, w)).astype(np.float32), sigma=max(h, w) / 12.0)
    noise = (noise - noise.min()) / (np.ptp(noise) + 1e-6)
    d = (1.0 - ramp) * strength + (1.0 - strength) * noise
    return np.clip(d, 0.0, 1.0)


def add_fog(img, rng):
    h, w = img.shape[:2]
    d = depth_proxy(h, w, rng)
    beta = rng.uniform(1.2, 3.2)
    t = np.exp(-beta * d)[..., None]
    airlight = rng.uniform(0.86, 1.0)
    return np.clip(img * t + airlight * (1.0 - t), 0.0, 1.0)


def add_dark(img, rng):
    gamma = rng.uniform(1.8, 2.8)
    scale = rng.uniform(0.30, 0.60)
    out = scale * np.power(np.clip(img, 0.0, 1.0), gamma)
    out += rng.normal(0.0, 0.02, out.shape).astype(np.float32)
    return np.clip(out, 0.0, 1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--voc", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--kind", choices=["fog", "dark"], default="fog")
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    rng = np.random.default_rng(a.seed)
    random.seed(a.seed)
    ann_dir = os.path.join(a.voc, "Annotations")
    img_dir = os.path.join(a.voc, "JPEGImages")
    ids = sorted(f[:-4] for f in os.listdir(ann_dir) if f.endswith(".xml"))
    random.shuffle(ids)
    ids = ids[: a.n]

    out_img = os.path.join(a.out, "JPEGImages")
    out_ann = os.path.join(a.out, "Annotations")
    os.makedirs(out_img, exist_ok=True)
    os.makedirs(out_ann, exist_ok=True)

    fn = add_fog if a.kind == "fog" else add_dark
    done = 0
    for stem in ids:
        src = os.path.join(img_dir, stem + ".jpg")
        if not os.path.exists(src):
            continue
        arr = np.asarray(Image.open(src).convert("RGB"), dtype=np.float32) / 255.0
        Image.fromarray((fn(arr, rng) * 255).astype(np.uint8)).save(
            os.path.join(out_img, stem + ".jpg"), quality=95
        )
        shutil.copy(os.path.join(ann_dir, stem + ".xml"), os.path.join(out_ann, stem + ".xml"))
        done += 1
    print(f"{a.kind}: wrote {done} images -> {a.out}")


if __name__ == "__main__":
    main()
