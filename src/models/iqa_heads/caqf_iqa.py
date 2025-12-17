"""CAQF_IQA: multi-scale fusion with content-adaptive weights and optional attention."""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..backbones.resnet_backbone import ResNetFeatureExtractor


class CAQF_IQA(nn.Module):
    def __init__(self, C: int = 256, use_attention: bool = True, mos_min: float = 0.0, mos_max: float = 100.0):
        super().__init__()
        self.backbone = ResNetFeatureExtractor("resnet50", pretrained=True)
        c2 = self.backbone.out_channels["layer2"]
        c3 = self.backbone.out_channels["layer3"]
        c4 = self.backbone.out_channels["layer4"]
        self.conv2 = nn.Conv2d(c2, C, kernel_size=1)
        self.conv3 = nn.Conv2d(c3, C, kernel_size=1)
        self.conv4 = nn.Conv2d(c4, C, kernel_size=1)
        self.mlp_w = nn.Sequential(
            nn.Linear(3 * C, C),
            nn.ReLU(inplace=True),
            nn.Linear(C, 3),
        )
        self.attn = nn.Conv2d(C, 1, kernel_size=1) if use_attention else None
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Sequential(
            nn.Linear(C, C // 2),
            nn.ReLU(inplace=True),
            nn.Linear(C // 2, 1),
        )
        self.mos_min = mos_min
        self.mos_max = mos_max

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)
        f2, f3, f4 = feats["layer2"], feats["layer3"], feats["layer4"]
        # Align to layer4 spatial size
        f2a = F.adaptive_avg_pool2d(f2, f4.shape[-2:])
        f3a = F.adaptive_avg_pool2d(f3, f4.shape[-2:])
        f2a = self.conv2(f2a)
        f3a = self.conv3(f3a)
        f4a = self.conv4(f4)
        # Global features
        g2 = self.pool(f2a).flatten(1)
        g3 = self.pool(f3a).flatten(1)
        g4 = self.pool(f4a).flatten(1)
        g = torch.cat([g2, g3, g4], dim=1)
        w = self.mlp_w(g)
        w = torch.softmax(w, dim=1)
        w2, w3, w4 = w[:, 0].view(-1, 1, 1, 1), w[:, 1].view(-1, 1, 1, 1), w[:, 2].view(-1, 1, 1, 1)
        fq = w2 * f2a + w3 * f3a + w4 * f4a
        if self.attn is not None:
            A = torch.sigmoid(self.attn(fq))
            fq = fq * A
        z = self.pool(fq).flatten(1)
        y = self.head(z)
        y = torch.tanh(y)
        y = (y + 1) * 0.5 * (self.mos_max - self.mos_min) + self.mos_min
        return y


class CAQF_IQA_NoAttn(CAQF_IQA):
    def __init__(self, C: int = 256, mos_min: float = 0.0, mos_max: float = 100.0):
        super().__init__(C=C, use_attention=False, mos_min=mos_min, mos_max=mos_max)

