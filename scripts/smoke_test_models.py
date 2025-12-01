import os
import sys
import torch

try:
    import torchvision
except Exception:
    print("torchvision not available; skipping")
    sys.exit(0)

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.models.backbones.resnet_backbone import ResNetBackbone50
from src.models.iqa_heads.caqf_iqa import CAQF_IQA
from src.models.iqa_heads.resnet_baseline import ResNetBaselineIQA

device = "cuda" if torch.cuda.is_available() else "cpu"
x = torch.randn(2, 3, 224, 224).to(device)

backbone = ResNetBackbone50(pretrained=False).to(device)
model_caqf = CAQF_IQA(backbone, out_range=(0.0, 100.0), C=256, spatial_attention=True).to(device)
y = model_caqf(x)
print("CAQF_IQA output", y.shape, float(y.min()), float(y.max()))

backbone2 = ResNetBackbone50(pretrained=False).to(device)
model_base = ResNetBaselineIQA(backbone2, out_range=(0.0, 100.0)).to(device)
y2 = model_base(x)
print("Baseline output", y2.shape, float(y2.min()), float(y2.max()))

