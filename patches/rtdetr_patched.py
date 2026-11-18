"""Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import torch 
import torch.nn as nn 
import torch.nn.functional as F 

import random 
import numpy as np 
from typing import List 

from ...core import register


__all__ = ['RTDETR', ]


@register()
class RTDETR(nn.Module):
    __inject__ = ['backbone', 'encoder', 'decoder', ]

    def __init__(self, \
        backbone: nn.Module, 
        encoder: nn.Module, 
        decoder: nn.Module, 
    ):
        super().__init__()
        self.backbone = backbone
        self.decoder = decoder
        self.encoder = encoder
        
    def forward(self, x, targets=None):
        x = self.backbone(x)
        x = self.encoder(x)        
        x = self.decoder(x, targets)

        return x
    
    def deploy(self, ):
        self.eval()
        for m in self.modules():
            if hasattr(m, 'convert_to_deploy'):
                m.convert_to_deploy()
        return self


from .freq_modules import DualDomainFrontend


@register()
class RTDETRFront(RTDETR):
    """RT-DETR with a frequency front-end applied to the input image."""

    def __init__(self, backbone: nn.Module, encoder: nn.Module, decoder: nn.Module,
                 front_hidden: int = 16, alpha: float = 0.5, beta: float = 0.1, gamma: float = 0.1,
                 use_wav: bool = True, use_fft: bool = True):
        super().__init__(backbone, encoder, decoder)
        self.front = DualDomainFrontend(3, hidden=front_hidden, alpha=alpha, beta=beta, gamma=gamma,
                                        use_wav=use_wav, use_fft=use_fft)

    def forward(self, x, targets=None):
        x = self.front(x)
        return super().forward(x, targets)
