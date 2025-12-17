"""Stair-IQA: coarse-to-fine fusion from layer4 → layer3 → layer2."""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..backbones.resnet_backbone import ResNetFeatureExtractor


class StairIQA(nn.Module):
    def __init__(self, C: int = 256, mos_min: float = 0.0, mos_max: float = 100.0):
        super().__init__()
        self.backbone = ResNetFeatureExtractor("resnet50", pretrained=True)
        c2 = self.backbone.out_channels["layer2"]
        c3 = self.backbone.out_channels["layer3"]
        c4 = self.backbone.out_channels["layer4"]
        self.proj2 = nn.Conv2d(c2, C, kernel_size=1)
        self.proj3 = nn.Conv2d(c3, C, kernel_size=1)
        self.proj4 = nn.Conv2d(c4, C, kernel_size=1)
        self.fuse3 = nn.Sequential(nn.Conv2d(2 * C, C, kernel_size=3, padding=1), nn.BatchNorm2d(C), nn.ReLU(inplace=True))
        self.fuse2 = nn.Sequential(nn.Conv2d(2 * C, C, kernel_size=3, padding=1), nn.BatchNorm2d(C), nn.ReLU(inplace=True))
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
        f2 = self.proj2(feats["layer2"])
        f3 = self.proj3(feats["layer3"])
        f4 = self.proj4(feats["layer4"])
        u4 = F.interpolate(f4, size=f3.shape[-2:], mode="bilinear", align_corners=False)
        s3 = self.fuse3(torch.cat([f3, u4], dim=1))
        u3 = F.interpolate(s3, size=f2.shape[-2:], mode="bilinear", align_corners=False)
        s2 = self.fuse2(torch.cat([f2, u3], dim=1))
        z = self.pool(s2).flatten(1)
        y = self.head(z)
        y = torch.tanh(y)
        y = (y + 1) * 0.5 * (self.mos_max - self.mos_min) + self.mos_min
        return y

