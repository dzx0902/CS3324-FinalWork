import torch
import torch.nn as nn
import torch.nn.functional as F
from ..backbones.resnet_backbone import ResNetBackbone50


class ResNetBaselineIQA(nn.Module):
    def __init__(self, backbone: ResNetBackbone50, out_range=(0.0, 100.0), Cmid: int = 512):
        super().__init__()
        self.backbone = backbone
        self.out_range = out_range
        in_dim = backbone.feat_dims["f4"]
        self.head = nn.Sequential(
            nn.Linear(in_dim, Cmid),
            nn.ReLU(inplace=True),
            nn.Linear(Cmid, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, f4 = self.backbone(x)
        z = F.adaptive_avg_pool2d(f4, 1).flatten(1)
        y_raw = self.head(z).squeeze(1)
        minv, maxv = self.out_range
        y = (torch.tanh(y_raw) + 1.0) * 0.5 * (maxv - minv) + minv
        return y

