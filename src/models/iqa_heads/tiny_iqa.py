"""TinyIQA_R18 and KD-ready variant."""
from __future__ import annotations
import torch
import torch.nn as nn
from ..backbones.resnet_backbone import ResNetFeatureExtractor


class TinyIQA_R18(nn.Module):
    def __init__(self, dropout: float = 0.0, mos_min: float = 0.0, mos_max: float = 100.0):
        super().__init__()
        self.backbone = ResNetFeatureExtractor("resnet18", pretrained=True)
        c4 = self.backbone.out_channels["layer4"]
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        head_layers = [nn.Linear(c4, 256), nn.ReLU(inplace=True)]
        if dropout > 0:
            head_layers.append(nn.Dropout(dropout))
        head_layers += [nn.Linear(256, 1)]
        self.head = nn.Sequential(*head_layers)
        self.mos_min = mos_min
        self.mos_max = mos_max

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f4 = self.backbone(x)["layer4"]
        z = self.pool(f4).flatten(1)
        y = self.head(z)
        y = torch.tanh(y)
        y = (y + 1) * 0.5 * (self.mos_max - self.mos_min) + self.mos_min
        return y

