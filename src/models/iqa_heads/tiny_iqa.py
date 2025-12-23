"""TinyIQA_R18 and KD-ready variant."""
from __future__ import annotations
import torch
import torch.nn as nn
from ..backbones.resnet_backbone import ResNetFeatureExtractor


class SEBlock(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid(),
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        y = self.avg(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class TinyIQA_R18(nn.Module):
    def __init__(self, dropout: float = 0.0, mos_min: float = 0.0, mos_max: float = 100.0, use_se: bool = False, hidden_dim1: int = 256, hidden_dim2: int = 0):
        super().__init__()
        self.backbone = ResNetFeatureExtractor("resnet18", pretrained=True)
        c4 = self.backbone.out_channels["layer4"]
        self.se = SEBlock(c4) if use_se else None
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        h1 = hidden_dim1 if hidden_dim1 > 0 else 256
        h2 = hidden_dim2 if hidden_dim2 > 0 else 0
        head_layers = [nn.Linear(c4, h1), nn.ReLU(inplace=True)]
        if dropout > 0:
            head_layers.append(nn.Dropout(dropout))
        if h2 > 0:
            head_layers += [nn.Linear(h1, h2), nn.ReLU(inplace=True)]
            head_layers += [nn.Linear(h2, 1)]
        else:
            head_layers += [nn.Linear(h1, 1)]
        self.head = nn.Sequential(*head_layers)
        self.mos_min = mos_min
        self.mos_max = mos_max

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f4 = self.backbone(x)["layer4"]
        if self.se is not None:
            f4 = self.se(f4)
        z = self.pool(f4).flatten(1)
        y = self.head(z)
        y = torch.tanh(y)
        y = (y + 1) * 0.5 * (self.mos_max - self.mos_min) + self.mos_min
        return y


class TinyIQA_R34(nn.Module):
    def __init__(self, dropout: float = 0.1, mos_min: float = 0.0, mos_max: float = 100.0, use_se: bool = False, hidden_dim1: int = 512, hidden_dim2: int = 0):
        super().__init__()
        self.backbone = ResNetFeatureExtractor("resnet34", pretrained=True)
        c4 = self.backbone.out_channels["layer4"]
        self.se = SEBlock(c4) if use_se else None
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        h1 = hidden_dim1 if hidden_dim1 > 0 else 512
        h2 = hidden_dim2 if hidden_dim2 > 0 else 0
        head_layers = [nn.Linear(c4, h1), nn.ReLU(inplace=True)]
        if dropout > 0:
            head_layers.append(nn.Dropout(dropout))
        if h2 > 0:
            head_layers += [nn.Linear(h1, h2), nn.ReLU(inplace=True)]
            head_layers += [nn.Linear(h2, 1)]
        else:
            head_layers += [nn.Linear(h1, 1)]
        self.head = nn.Sequential(*head_layers)
        self.mos_min = mos_min
        self.mos_max = mos_max

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f4 = self.backbone(x)["layer4"]
        if self.se is not None:
            f4 = self.se(f4)
        z = self.pool(f4).flatten(1)
        y = self.head(z)
        y = torch.tanh(y)
        y = (y + 1) * 0.5 * (self.mos_max - self.mos_min) + self.mos_min
        return y


class TinyIQA_R18_MS(nn.Module):
    def __init__(self, C: int = 256, dropout: float = 0.1, mos_min: float = 0.0, mos_max: float = 100.0):
        super().__init__()
        self.backbone = ResNetFeatureExtractor("resnet18", pretrained=True)
        c3 = self.backbone.out_channels["layer3"]
        c4 = self.backbone.out_channels["layer4"]
        self.conv3 = nn.Conv2d(c3, C, kernel_size=1)
        self.conv4 = nn.Conv2d(c4, C, kernel_size=1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.mlp_w = nn.Sequential(nn.Linear(2 * C, C), nn.ReLU(inplace=True), nn.Linear(C, 2))
        head = [nn.Linear(C, C // 2), nn.ReLU(inplace=True)]
        if dropout > 0:
            head.append(nn.Dropout(dropout))
        head += [nn.Linear(C // 2, 1)]
        self.head = nn.Sequential(*head)
        self.mos_min = mos_min
        self.mos_max = mos_max

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)
        f3 = self.conv3(feats["layer3"])
        f4 = self.conv4(feats["layer4"])
        f3u = torch.nn.functional.interpolate(f3, size=f4.shape[-2:], mode="bilinear", align_corners=False)
        g3 = self.pool(f3u).flatten(1)
        g4 = self.pool(f4).flatten(1)
        g = torch.cat([g3, g4], dim=1)
        w = torch.softmax(self.mlp_w(g), dim=1)
        w3, w4 = w[:, 0].view(-1, 1, 1, 1), w[:, 1].view(-1, 1, 1, 1)
        fq = w3 * f3u + w4 * f4
        z = self.pool(fq).flatten(1)
        y = self.head(z)
        y = torch.tanh(y)
        y = (y + 1) * 0.5 * (self.mos_max - self.mos_min) + self.mos_min
        return y

