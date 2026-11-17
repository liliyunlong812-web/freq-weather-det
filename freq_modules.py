"""Frequency modules for adverse-weather object detection.

C1  Dual-domain Frequency Decoupling (DFD): Haar DWT sub-band gating + FFT magnitude/phase split.
C2  Detection-Evidence Guided Selection (DEGS): spatial gate from a shallow detector feature.
C3  Phase-consistency loss and offline logit distillation loss.

The modules are detector-agnostic. In this project they sit between the input image and the
RT-DETR backbone; DEGS consumes the stride-8 feature produced by the first backbone stage.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "HaarDWT",
    "HaarIDWT",
    "DualDomainFrontend",
    "DetectionEvidenceGate",
    "phase_consistency_loss",
    "logit_distill_loss",
    "topk_soft_mask",
]


class HaarDWT(nn.Module):
    """One-level, channel-wise Haar decomposition. Returns (LL, LH, HL, HH)."""

    def __init__(self, channels: int):
        super().__init__()
        k = torch.tensor(
            [
                [[1.0, 1.0], [1.0, 1.0]],
                [[1.0, 1.0], [-1.0, -1.0]],
                [[1.0, -1.0], [1.0, -1.0]],
                [[1.0, -1.0], [-1.0, 1.0]],
            ]
        ) / 4.0
        weight = k.repeat(channels, 1, 1, 1)  # (4C, 1, 2, 2)
        self.register_buffer("weight", weight)
        self.channels = channels

    def forward(self, x: torch.Tensor):
        b, c, h, w = x.shape
        if h % 2 or w % 2:
            x = F.pad(x, (0, w % 2, 0, h % 2), mode="replicate")
        out = F.conv2d(x, self.weight, stride=2, groups=c)
        out = out.view(b, 4, c, out.shape[-2], out.shape[-1])
        return out[:, 0], out[:, 1], out[:, 2], out[:, 3]


class HaarIDWT(nn.Module):
    """Inverse of HaarDWT. Restores the spatial resolution from four sub-bands."""

    def __init__(self, channels: int):
        super().__init__()
        k = torch.tensor(
            [
                [[1.0, 1.0], [1.0, 1.0]],
                [[1.0, 1.0], [-1.0, -1.0]],
                [[1.0, -1.0], [1.0, -1.0]],
                [[1.0, -1.0], [-1.0, 1.0]],
            ]
        ) / 4.0
        weight = k.repeat(channels, 1, 1, 1)
        self.register_buffer("weight", weight)
        self.channels = channels

    def forward(self, ll, lh, hl, hh):
        b, c, h, w = ll.shape
        stack = torch.stack([ll, lh, hl, hh], dim=1).view(b, 4 * c, h, w)
        return F.conv_transpose2d(stack, self.weight, stride=2, groups=c)


class SubBandGate(nn.Module):
    """Per-direction, per-channel, spatially varying gate for one wavelet sub-band."""

    def __init__(self, channels: int, hidden: int = 16):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, hidden, 3, padding=1, groups=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, channels, 1),
            nn.Sigmoid(),
        )
        nn.init.zeros_(self.body[-2].weight)
        nn.init.constant_(self.body[-2].bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # residual gate: identity at init (sigmoid(0) = 0.5 -> 2*g - 1 = 0)
        return x * (2.0 * self.body(x) - 1.0)


class DualDomainFrontend(nn.Module):
    """C1: wavelet sub-band correction plus FFT magnitude gating with phase preservation."""

    def __init__(self, channels: int = 3, hidden: int = 16, alpha: float = 0.5, beta: float = 0.1, gamma: float = 0.1):
        super().__init__()
        self.dwt = HaarDWT(channels)
        self.idwt = HaarIDWT(channels)
        self.gate_lh = SubBandGate(channels, hidden)
        self.gate_hl = SubBandGate(channels, hidden)
        self.gate_hh = SubBandGate(channels, hidden)
        self.amp = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, channels, 1),
            nn.Tanh(),
        )
        self.pha = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, channels, 1),
            nn.Tanh(),
        )
        nn.init.zeros_(self.amp[-2].weight)
        nn.init.zeros_(self.amp[-2].bias)
        nn.init.zeros_(self.pha[-2].weight)
        nn.init.zeros_(self.pha[-2].bias)
        self.alpha, self.beta, self.gamma = alpha, beta, gamma

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ---- wavelet branch: correct directional high-frequency sub-bands, keep LL ----
        ll, lh, hl, hh = self.dwt(x)
        wav = self.idwt(ll, self.gate_lh(lh), self.gate_hl(hl), self.gate_hh(hh))
        wav = wav[..., : x.shape[-2], : x.shape[-1]]

        # ---- Fourier branch: gate magnitude, keep phase with a small residual ----
        spec = torch.fft.rfft2(x, norm="ortho")
        mag, pha = spec.abs(), spec.angle()
        mag = mag * (1.0 + self.alpha * self.amp(mag))
        pha = pha + self.beta * self.pha(pha)
        rec = torch.fft.irfft2(torch.polar(mag, pha), s=x.shape[-2:], norm="ortho")

        return x + self.gamma * (0.5 * (wav + rec))


class DetectionEvidenceGate(nn.Module):
    """C2: spatial gate derived from a shallow detector feature map."""

    def __init__(self, in_channels: int, hidden: int = 32):
        super().__init__()
        self.reduce = nn.Sequential(
            nn.Conv2d(in_channels, hidden, 1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, 1, 1),
        )

    def forward(self, feat: torch.Tensor, size, warmup: bool = False, topk: float = 0.15):
        m = self.reduce(feat)
        m = F.interpolate(m, size=size, mode="bilinear", align_corners=False)
        m = torch.sigmoid(m)
        if warmup:
            m = topk_soft_mask(m, topk)
        return m


def topk_soft_mask(mask: torch.Tensor, ratio: float) -> torch.Tensor:
    """Keep the strongest `ratio` fraction of responses, suppress the rest softly."""
    b, _, h, w = mask.shape
    flat = mask.view(b, -1)
    k = max(1, int(flat.shape[1] * ratio))
    thresh = flat.topk(k, dim=1).values[:, -1].view(b, 1, 1, 1)
    return mask * torch.sigmoid((mask - thresh) * 20.0)


def phase_consistency_loss(enhanced: torch.Tensor, degraded: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """C3-a: penalise phase drift introduced by the frontend."""
    fe = torch.fft.rfft2(enhanced, norm="ortho")
    fd = torch.fft.rfft2(degraded, norm="ortho")
    cos = (fe.real * fd.real + fe.imag * fd.imag) / (fe.abs() * fd.abs() + eps)
    return (1.0 - cos.clamp(-1.0, 1.0)).mean()


def logit_distill_loss(student_logits, teacher_logits, temperature: float = 2.0) -> torch.Tensor:
    """C3-b: KL divergence from a frozen clean-domain teacher (logits cached offline)."""
    s = F.log_softmax(student_logits / temperature, dim=-1)
    t = F.softmax(teacher_logits / temperature, dim=-1)
    return F.kl_div(s, t, reduction="batchmean") * (temperature ** 2)


if __name__ == "__main__":
    x = torch.randn(2, 3, 64, 64)
    front = DualDomainFrontend(3)
    y = front(x)
    print("frontend out", tuple(y.shape), "params", sum(p.numel() for p in front.parameters()))
    gate = DetectionEvidenceGate(64)
    g = gate(torch.randn(2, 64, 8, 8), size=(64, 64), warmup=True)
    print("gate out", tuple(g.shape), g.min().item(), g.max().item())
    print("phase loss", float(phase_consistency_loss(y, x)))
