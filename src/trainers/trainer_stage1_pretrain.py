"""Stage-1 synthetic distortion classification pretraining."""
from __future__ import annotations
import os
from typing import List
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import torchvision.transforms as T
from tqdm import tqdm
from ..datasets.distortion_augment import random_distortion
from ..models.backbones.resnet_backbone import ResNetFeatureExtractor
from ..utils.checkpoint_utils import save_ckpt


class SyntheticDistortionDataset(Dataset):
    def __init__(self, image_paths: List[str]):
        self.paths = image_paths
        self.t_basic = T.Compose([T.Resize((224, 224)), T.ToTensor(),
                                  T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        path = self.paths[idx]
        img = Image.open(path).convert("RGB")
        img_dist = random_distortion(img)
        x_clean = self.t_basic(img)
        x_dist = self.t_basic(img_dist)
        return x_clean, torch.tensor(0, dtype=torch.long), x_dist, torch.tensor(1, dtype=torch.long)


class DistortionClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = ResNetFeatureExtractor("resnet50", pretrained=True)
        c4 = self.backbone.out_channels["layer4"]
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(c4, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f4 = self.backbone(x)["layer4"]
        z = self.pool(f4).flatten(1)
        return self.fc(z)


def pretrain(image_root: str, file_list_txt: str, out_ckpt: str, epochs: int = 1, batch_size: int = 16, device: str = "cuda"):
    with open(file_list_txt, "r", encoding="utf-8") as f:
        rels = [line.strip() for line in f if line.strip()]
    paths = [os.path.join(image_root, r) for r in rels]
    ds = SyntheticDistortionDataset(paths)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=4)
    device_t = torch.device(device if torch.cuda.is_available() else "cpu")
    model = DistortionClassifier().to(device_t)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    ce = nn.CrossEntropyLoss()
    for epoch in range(1, epochs + 1):
        pbar = tqdm(loader, desc=f"Stage-1 Epoch {epoch}", leave=False)
        for x0, y0, x1, y1 in pbar:
            x = torch.cat([x0, x1], dim=0).to(device_t)
            y = torch.cat([y0, y1], dim=0).to(device_t)
            logits = model(x)
            loss = ce(logits, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            pbar.set_postfix(loss=f"{loss.item():.4f}")
    save_ckpt(model, opt, epochs, out_ckpt)

