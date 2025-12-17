"""AGIQA-3K dataset loader reading CSV with image paths and MOS."""
from __future__ import annotations
import csv
import os
from typing import List, Tuple
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T


class AGIQADataset(Dataset):
    def __init__(self, image_root: str, meta_csv: str, transform: T.Compose | None = None):
        self.image_root = image_root
        self.records: List[Tuple[str, float]] = []
        with open(meta_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.records.append((row["path"], float(row["mos"])))
        self.transform = transform or T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        rel, mos = self.records[idx]
        path = os.path.join(self.image_root, rel)
        img = Image.open(path).convert("RGB")
        img_t = self.transform(img)
        return img_t, torch.tensor([mos], dtype=torch.float32)

