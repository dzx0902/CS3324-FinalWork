import torch
import torch.nn as nn
import torch.nn.functional as F
from ..backbones.resnet_backbone import ResNetBackbone50


class CAQF_IQA(nn.Module):
    def __init__(self, backbone: ResNetBackbone50, out_range=(0.0, 100.0), C: int = 256, spatial_attention: bool = True):
        super().__init__()
        self.backbone = backbone
        self.out_range = out_range
        self.proj2 = nn.Conv2d(backbone.feat_dims["f2"], C, kernel_size=1)
        self.proj3 = nn.Conv2d(backbone.feat_dims["f3"], C, kernel_size=1)
        self.proj4 = nn.Conv2d(backbone.feat_dims["f4"], C, kernel_size=1)
        self.weight_mlp = nn.Sequential(
            nn.Linear(C * 3, C // 2),
            nn.ReLU(inplace=True),
            nn.Linear(C // 2, 3),
        )
        self.spatial_attention = spatial_attention
        if spatial_attention:
            self.attn_conv = nn.Conv2d(C, 1, kernel_size=1)
        self.reg_head = nn.Sequential(
            nn.Linear(C, C // 2),
            nn.ReLU(inplace=True),
            nn.Linear(C // 2, 1),
        )

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        f2, f3, f4 = self.backbone(x)
        out_h, out_w = f4.shape[-2], f4.shape[-1]
        f2a = F.adaptive_avg_pool2d(f2, (out_h, out_w))
        f3a = F.adaptive_avg_pool2d(f3, (out_h, out_w))
        f2a = self.proj2(f2a)
        f3a = self.proj3(f3a)
        f4a = self.proj4(f4)
        g2 = F.adaptive_avg_pool2d(f2a, 1).flatten(1)
        g3 = F.adaptive_avg_pool2d(f3a, 1).flatten(1)
        g4 = F.adaptive_avg_pool2d(f4a, 1).flatten(1)
        g = torch.cat([g2, g3, g4], dim=1)
        w = self.weight_mlp(g)
        w = F.softmax(w, dim=1)
        w2 = w[:, 0].view(-1, 1, 1, 1)
        w3 = w[:, 1].view(-1, 1, 1, 1)
        w4 = w[:, 2].view(-1, 1, 1, 1)
        fq = w2 * f2a + w3 * f3a + w4 * f4a
        if self.spatial_attention:
            A = torch.sigmoid(self.attn_conv(fq))
            fq = fq * A
        z = F.adaptive_avg_pool2d(fq, 1).flatten(1)
        return z

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.forward_features(x)
        y_raw = self.reg_head(z).squeeze(1)
        minv, maxv = self.out_range
        y = (torch.tanh(y_raw) + 1.0) * 0.5 * (maxv - minv) + minv
        return y

