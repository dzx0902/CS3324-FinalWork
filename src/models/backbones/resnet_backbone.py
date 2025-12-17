"""ResNet feature extractor backbone returning intermediate feature maps."""
from __future__ import annotations
from typing import Dict, Tuple
import torch
import torch.nn as nn
import torchvision.models as tvm


class ResNetFeatureExtractor(nn.Module):
    def __init__(self, name: str = "resnet50", pretrained: bool = True):
        super().__init__()
        if name == "resnet50":
            net = tvm.resnet50(weights=tvm.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
            c_out = 2048
        elif name == "resnet18":
            net = tvm.resnet18(weights=tvm.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
            c_out = 512
        else:
            raise ValueError(f"Unsupported backbone {name}")
        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool)
        self.layer1 = net.layer1
        self.layer2 = net.layer2
        self.layer3 = net.layer3
        self.layer4 = net.layer4
        self.out_channels = {
            "layer1": list(self.layer1.modules())[-1].bn2.num_features if name != "resnet18" else 64,
            "layer2": list(self.layer2.modules())[-1].bn2.num_features if name != "resnet18" else 128,
            "layer3": list(self.layer3.modules())[-1].bn2.num_features if name != "resnet18" else 256,
            "layer4": c_out,
        }

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        x = self.stem(x)
        f1 = self.layer1(x)
        f2 = self.layer2(f1)
        f3 = self.layer3(f2)
        f4 = self.layer4(f3)
        return {"layer1": f1, "layer2": f2, "layer3": f3, "layer4": f4}

