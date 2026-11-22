"""Post-hoc analysis: is the front-end doing anything beyond an unsharp mask?

Loads the trained C1-64 model, hooks the front-end, and for 200 validation images computes
  (a) cosine similarity between the correction (x' - x) and a fixed unsharp mask,
  (b) the radial gain |F(x')| / |F(x)| in eight frequency bands,
  (c) the directional gain in four orientation sectors of the high-frequency band.
"""

import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.ndimage import gaussian_filter

sys.path.insert(0, '.')
from src.core import YAMLConfig

cfg = YAMLConfig('configs/rtdetrv2/rtdetrv2_r18vd_fog_c1L.yml')
model = cfg.model.eval().cuda()
state = torch.load('/home/ubuntu/ws/runs/L_0/best.pth', map_location='cpu', weights_only=False)
model.load_state_dict(state['model'], strict=False)

captured = {}
front = model.front


def hook(mod, inp, out):
    captured['x'] = inp[0].detach()
    captured['y'] = out.detach()


front.register_forward_hook(hook)

val = json.load(open('/home/ubuntu/ws/data/fog_voc/annotations/val.json'))
root = '/home/ubuntu/ws/data/fog_voc/JPEGImages'
files = [it['file_name'] for it in val['images']][:200]

cosines, radial, directional = [], np.zeros(8), np.zeros(4)
band_edges = np.array([0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 1.0])

for fname in files:
    img = Image.open(os.path.join(root, fname)).convert('RGB').resize((640, 640))
    x = torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0).permute(2, 0, 1)[None].cuda()
    with torch.no_grad():
        front(x)
    x0, y0 = captured['x'], captured['y']
    r = (y0 - x0)[0].permute(1, 2, 0).cpu().numpy()
    x_np = x0[0].permute(1, 2, 0).cpu().numpy()
    u = x_np - np.stack([gaussian_filter(x_np[..., c], 1.5) for c in range(3)], -1)
    a, b = r.ravel(), u.ravel()
    cosines.append(float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)))

    # radial / directional gain
    gx = np.fft.fftshift(np.abs(np.fft.fft2(x_np.mean(2))))
    gy = np.fft.fftshift(np.abs(np.fft.fft2(y0[0].mean(0).cpu().numpy())))
    h, w = gx.shape
    yy, xx = np.mgrid[0:h, 0:w]
    cy, cx = h // 2, w // 2
    rad = np.hypot((yy - cy) / (h / 2), (xx - cx) / (w / 2))
    gain = gy / (gx + 1e-6)
    for i in range(8):
        m = (rad >= band_edges[i]) & (rad < band_edges[i + 1])
        if m.any():
            radial[i] += float(gain[m].mean())
    ang = (np.degrees(np.arctan2(yy - cy, xx - cx)) + 180) % 180
    hf = rad > 0.5
    for i in range(4):
        m = hf & (ang >= i * 45) & (ang < (i + 1) * 45)
        if m.any():
            directional[i] += float(gain[m].mean())

n = len(files)
radial /= n
directional /= n

out = {
    'n_images': n,
    'cosine_vs_unsharp_mean': float(np.mean(cosines)),
    'cosine_vs_unsharp_std': float(np.std(cosines)),
    'radial_gain': [round(float(v), 3) for v in radial],
    'radial_band_edges': band_edges.tolist(),
    'directional_gain_highfreq': [round(float(v), 3) for v in directional],
}
json.dump(out, open('/home/ubuntu/ws/logs/probe_mechanism.json', 'w'), indent=2)
print(json.dumps(out, indent=2))
