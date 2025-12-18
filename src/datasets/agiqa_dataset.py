"""AGIQA-3K dataset loader supporting JSON/CSV metas."""
from __future__ import annotations
import os
from typing import List, Tuple
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
from .meta_utils import load_meta


class AGIQADataset(Dataset):
    def __init__(self, image_root: str, meta_path: str, transform: T.Compose | None = None):
        self.image_root = image_root
        self.records: List[Tuple[str, float]] = load_meta(meta_path, preferred_mos_key="MOS_zscore")
        self.transform = transform or T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        rel, mos = self.records[idx]
        path = rel if os.path.isabs(rel) else os.path.join(self.image_root, rel)
        img = Image.open(path).convert("RGB")
        img_t = self.transform(img)
        return img_t, torch.tensor([float(mos)], dtype=torch.float32)
