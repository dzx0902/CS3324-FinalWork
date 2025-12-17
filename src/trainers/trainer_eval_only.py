"""Evaluation utilities and prediction vs MOS scatter plotting."""
from __future__ import annotations
import os
from typing import Tuple, List
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from ..utils.metrics import plcc, srcc, rmse, mae
from ..datasets.koniq_dataset import KonIQDataset
from ..datasets.spaq_dataset import SPAQDataset
from ..datasets.kadid_dataset import KADIDDataset
from ..datasets.agiqa_dataset import AGIQADataset


def build_loader(dataset_name: str, image_root: str, meta_path: str, batch_size: int = 64, num_workers: int = 4) -> DataLoader:
    dn = dataset_name.lower()
    if dn == "koniq":
        ds = KonIQDataset(image_root, meta_path)
    elif dn == "spaq":
        ds = SPAQDataset(image_root, meta_path)
    elif dn == "kadid":
        ds = KADIDDataset(image_root, meta_path)
    elif dn == "agiqa":
        ds = AGIQADataset(image_root, meta_path)
    else:
        raise ValueError(f"Unknown dataset {dataset_name}")
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)


def eval_model(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> Tuple[float, float, float, float, List[float], List[float]]:
    model.eval()
    all_y = []
    all_p = []
    with torch.no_grad():
        for imgs, mos in loader:
            imgs = imgs.to(device)
            mos = mos.to(device)
            pred = model(imgs)
            all_y.extend(mos.view(-1).cpu().numpy().tolist())
            all_p.extend(pred.view(-1).cpu().numpy().tolist())
    y = np.array(all_y)
    p = np.array(all_p)
    return plcc(y, p), srcc(y, p), rmse(y, p), mae(y, p), all_y, all_p


def plot_pred_vs_mos(y_true: List[float], y_pred: List[float], out_path: str, title: str = "") -> None:
    plt.figure()
    plt.scatter(y_true, y_pred, alpha=0.5, s=10)
    min_v = min(min(y_true), min(y_pred))
    max_v = max(max(y_true), max(y_pred))
    plt.plot([min_v, max_v], [min_v, max_v], "r--", label="y=x")
    plt.xlabel("MOS (ground truth)")
    plt.ylabel("Predicted MOS")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path)
    plt.close()

