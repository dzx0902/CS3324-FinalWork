import torch
import torch.nn as nn
import torch.nn.functional as F
from .backbones.resnet_backbone import ResNetBackbone50


class StairIQA(nn.Module):
    def __init__(self, backbone: ResNetBackbone50, out_range=(0.0, 100.0), C: int = 256):
        super().__init__()
        self.backbone = backbone
        self.out_range = out_range
        self.p2 = nn.Conv2d(backbone.feat_dims["f2"], C, kernel_size=1)
        self.p3 = nn.Conv2d(backbone.feat_dims["f3"], C, kernel_size=1)
        self.p4 = nn.Conv2d(backbone.feat_dims["f4"], C, kernel_size=1)
        self.head = nn.Sequential(
            nn.Linear(C * 3, C),
            nn.ReLU(inplace=True),
            nn.Linear(C, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f2, f3, f4 = self.backbone(x)
        g2 = F.adaptive_avg_pool2d(self.p2(f2), 1).flatten(1)
        g3 = F.adaptive_avg_pool2d(self.p3(f3), 1).flatten(1)
        g4 = F.adaptive_avg_pool2d(self.p4(f4), 1).flatten(1)
        z = torch.cat([g2, g3, g4], dim=1)
        y_raw = self.head(z).squeeze(1)
        minv, maxv = self.out_range
        y = (torch.tanh(y_raw) + 1.0) * 0.5 * (maxv - minv) + minv
        return y

