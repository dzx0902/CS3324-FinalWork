"""HyperIQA: hyper-network generates adaptive scaling for high-level channels."""
from __future__ import annotations
import torch
import torch.nn as nn
from ..backbones.resnet_backbone import ResNetFeatureExtractor


class HyperIQA(nn.Module):
    def __init__(self, hidden: int = 256, mos_min: float = 0.0, mos_max: float = 100.0):
        super().__init__()
        self.backbone = ResNetFeatureExtractor("resnet50", pretrained=True)
        c4 = self.backbone.out_channels["layer4"]
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.hyper_mlp = nn.Sequential(
            nn.Linear(c4, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, c4),
            nn.Sigmoid(),
        )
        self.head = nn.Sequential(
            nn.Linear(c4, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 1),
        )
        self.mos_min = mos_min
        self.mos_max = mos_max

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f4 = self.backbone(x)["layer4"]
        g = self.pool(f4).flatten(1)
        scale = self.hyper_mlp(g).unsqueeze(-1).unsqueeze(-1)
        f4_adapt = f4 * scale
        z = self.pool(f4_adapt).flatten(1)
        y = self.head(z)
        y = torch.tanh(y)
        y = (y + 1) * 0.5 * (self.mos_max - self.mos_min) + self.mos_min
        return y

