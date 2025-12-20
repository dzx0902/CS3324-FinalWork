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
            try:
                net = tvm.resnet50(weights=tvm.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
            except Exception:
                net = tvm.resnet50(pretrained=pretrained)
        elif name == "resnet18":
            try:
                net = tvm.resnet18(weights=tvm.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
            except Exception:
                net = tvm.resnet18(pretrained=pretrained)
        elif name == "resnet34":
            try:
                net = tvm.resnet34(weights=tvm.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None)
            except Exception:
                net = tvm.resnet34(pretrained=pretrained)
        else:
            raise ValueError(f"Unsupported backbone {name}")
        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool)
        self.layer1 = net.layer1
        self.layer2 = net.layer2
        self.layer3 = net.layer3
        self.layer4 = net.layer4
        if name == "resnet50":
            self.out_channels = {"layer1": 256, "layer2": 512, "layer3": 1024, "layer4": 2048}
        elif name == "resnet34":
            self.out_channels = {"layer1": 64, "layer2": 128, "layer3": 256, "layer4": 512}
        else:
            self.out_channels = {"layer1": 64, "layer2": 128, "layer3": 256, "layer4": 512}

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        x = self.stem(x)
        f1 = self.layer1(x)
        f2 = self.layer2(f1)
        f3 = self.layer3(f2)
        f4 = self.layer4(f3)
        return {"layer1": f1, "layer2": f2, "layer3": f3, "layer4": f4}

