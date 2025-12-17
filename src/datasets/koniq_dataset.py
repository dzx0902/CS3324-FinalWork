"""KonIQ-10k dataset for NR-IQA with MOS labels."""
from __future__ import annotations
import json
import os
from typing import List, Dict, Tuple
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T


class KonIQDataset(Dataset):
    def __init__(self, image_root: str, split_meta: str, transform: T.Compose | None = None):
        self.image_root = image_root
        with open(split_meta, "r", encoding="utf-8") as f:
            self.items: List[Dict] = json.load(f)
        self.transform = transform or T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        it = self.items[idx]
        path = os.path.join(self.image_root, it["path"])
        mos = float(it["mos"])
        img = Image.open(path).convert("RGB")
        img_t = self.transform(img)
        return img_t, torch.tensor([mos], dtype=torch.float32)

