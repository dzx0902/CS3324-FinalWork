import csv
import json
import os
import random
from typing import List, Tuple, Optional

import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

from .distortion_augment import (
    add_gaussian_noise,
    add_gaussian_blur,
    jpeg_compress,
    adjust_brightness_contrast,
)


class KonIQDistortionClsDataset(Dataset):
    def __init__(
        self,
        root: str,
        csv_path: Optional[str] = None,
        image_size: int = 224,
        include_clean: bool = True,
        gaussian_noise_levels: List[float] = [5.0, 10.0, 20.0],
        gaussian_blur_sigmas: List[float] = [0.5, 1.0, 2.0],
        jpeg_qs: List[int] = [90, 70, 50],
        bc_params: List[Tuple[float, float]] = [(0.0, 1.0), (0.1, 1.1), (-0.1, 0.9)],
        clean_mos_threshold: Optional[float] = None,
    ):
        self.root = root
        self.include_clean = include_clean
        self.image_paths = []
        self.mos = {}
        if csv_path and os.path.isfile(csv_path):
            with open(csv_path, "r", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    p = os.path.join(root, row["image_name"]) if "image_name" in row else os.path.join(root, row["path"]) 
                    self.image_paths.append(p)
                    if "mos" in row:
                        self.mos[p] = float(row["mos"])
        else:
            for fn in os.listdir(root):
                if fn.lower().endswith((".jpg", ".jpeg", ".png")):
                    self.image_paths.append(os.path.join(root, fn))
        if clean_mos_threshold is not None and self.mos:
            self.image_paths = [p for p in self.image_paths if self.mos.get(p, 0.0) >= clean_mos_threshold]

        self.noise_levels = gaussian_noise_levels
        self.blur_sigmas = gaussian_blur_sigmas
        self.jpeg_qs = jpeg_qs
        self.bc_params = bc_params

        self.transforms = T.Compose(
            [
                T.Resize(image_size),
                T.RandomCrop(image_size),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
            ]
        )

        self.distort_types = [
            "clean",
            "gaussian_noise",
            "gaussian_blur",
            "jpeg",
            "brightness_contrast",
        ]

        self.class_map = []
        if include_clean:
            self.class_map.append(("clean", None))
        for s in self.noise_levels:
            self.class_map.append(("gaussian_noise", s))
        for s in self.blur_sigmas:
            self.class_map.append(("gaussian_blur", s))
        for q in self.jpeg_qs:
            self.class_map.append(("jpeg", q))
        for bc in self.bc_params:
            self.class_map.append(("brightness_contrast", bc))

    def __len__(self):
        return len(self.image_paths)

    def _apply_distortion(self, img: Image.Image) -> Tuple[Image.Image, int]:
        idx = random.randrange(len(self.class_map))
        name, param = self.class_map[idx]
        if name == "clean":
            return img.convert("RGB"), idx
        if name == "gaussian_noise":
            return add_gaussian_noise(img.convert("RGB"), float(param)), idx
        if name == "gaussian_blur":
            return add_gaussian_blur(img.convert("RGB"), 3, float(param)), idx
        if name == "jpeg":
            return jpeg_compress(img.convert("RGB"), int(param)), idx
        if name == "brightness_contrast":
            b, c = param
            return adjust_brightness_contrast(img.convert("RGB"), float(b), float(c)), idx
        return img.convert("RGB"), idx

    def __getitem__(self, i: int):
        p = self.image_paths[i]
        img = Image.open(p).convert("RGB")
        img_d, cid = self._apply_distortion(img)
        x = self.transforms(img_d)
        return x, cid


class KonIQMOSDataset(Dataset):
    def __init__(
        self,
        root: str,
        csv_path: Optional[str] = None,
        split: str = "train",
        image_size: int = 224,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        seed: int = 42,
        file_list_json: Optional[str] = None,
    ):
        self.root = root
        # 构建 MOS 映射
        mos_map = {}
        if csv_path and os.path.isfile(csv_path):
            with open(csv_path, "r", newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    name = row.get("image_name") or row.get("path") or row.get("name")
                    if not name:
                        continue
                    mos_val = row.get("mos") or row.get("MOS") or row.get("label")
                    mos_map[str(name)] = float(mos_val)
        # 构建文件列表
        file_list = None
        if file_list_json and os.path.isfile(file_list_json):
            with open(file_list_json, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict) and "images" in data:
                    file_list = list(map(str, data["images"]))
                elif isinstance(data, list):
                    file_list = list(map(str, data))
        items = []
        if file_list:
            for name in file_list:
                p = os.path.join(root, name)
                mos = mos_map.get(name)
                if mos is None:
                    # 兼容 root 中直接查找 CSV 行（若 CSV 提供路径）
                    mos = mos_map.get(os.path.basename(name), None)
                if mos is None:
                    continue
                items.append((p, mos))
        else:
            # 回退：仅使用 CSV 构造
            for name, mos in mos_map.items():
                p = os.path.join(root, name)
                items.append((p, mos))
        random.Random(seed).shuffle(items)
        n = len(items)
        n_test = int(n * test_ratio)
        n_val = int(n * val_ratio)
        test_set = items[:n_test]
        val_set = items[n_test : n_test + n_val]
        train_set = items[n_test + n_val :]
        if split == "train":
            self.items = train_set
            self.train_mode = True
        elif split == "val":
            self.items = val_set
            self.train_mode = False
        else:
            # 若提供了专门的测试列表，可将 test_ratio 设为 1.0 并通过 file_list_json 指定
            self.items = test_set if n_test > 0 else items
            self.train_mode = False

        self.transforms_train = T.Compose(
            [
                T.Resize(image_size),
                T.RandomCrop(image_size),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
            ]
        )
        self.transforms_eval = T.Compose([T.Resize(image_size), T.CenterCrop(image_size), T.ToTensor()])

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i: int):
        p, mos = self.items[i]
        img = Image.open(p).convert("RGB")
        if self.train_mode:
            x = self.transforms_train(img)
        else:
            x = self.transforms_eval(img)
        y = torch.tensor(mos, dtype=torch.float32)
        return x, y

