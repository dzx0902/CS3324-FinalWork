import torch
import torch.nn as nn
import torch.nn.functional as F
from .backbones.resnet_backbone import ResNetBackbone50


class HyperIQA(nn.Module):
    def __init__(self, backbone: ResNetBackbone50, out_range=(0.0, 100.0)):
        super().__init__()
        self.backbone = backbone
        self.out_range = out_range
        in_dim = backbone.feat_dims["f4"]
        self.hyper = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 2 * in_dim),
        )
        self.head = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, f4 = self.backbone(x)
        g = F.adaptive_avg_pool2d(f4, 1).flatten(1)
        hb = self.hyper(g)
        in_dim = f4.shape[1]
        gamma = hb[:, :in_dim].unsqueeze(-1).unsqueeze(-1)
        beta = hb[:, in_dim:].unsqueeze(-1).unsqueeze(-1)
        f4m = gamma * f4 + beta
        z = F.adaptive_avg_pool2d(f4m, 1).flatten(1)
        y_raw = self.head(z).squeeze(1)
        minv, maxv = self.out_range
        y = (torch.tanh(y_raw) + 1.0) * 0.5 * (maxv - minv) + minv
        return y

