"""Stage-2 MOS regression trainer with tqdm and history logging."""
from __future__ import annotations
import os
import json
from typing import Tuple, List
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from ..utils.seed_utils import set_global_seed
from ..utils.config_utils import load_yaml_config, build_train_config
from ..utils.logging_utils import save_history, plot_training_curves
from ..utils.metrics import plcc, srcc
from ..utils.checkpoint_utils import save_ckpt, load_model_ckpt
from ..losses.regression_losses import build_loss
from ..datasets.koniq_dataset import KonIQDataset
from ..datasets.spaq_dataset import SPAQDataset
from ..datasets.kadid_dataset import KADIDDataset
from ..datasets.agiqa_dataset import AGIQADataset
from ..models.iqa_heads.resnet_baseline import ResNetBaselineIQA
from ..models.iqa_heads.stair_iqa import StairIQA
from ..models.iqa_heads.hyper_iqa import HyperIQA
from ..models.iqa_heads.caqf_iqa import CAQF_IQA, CAQF_IQA_NoAttn
from ..models.iqa_heads.tiny_iqa import TinyIQA_R18


def build_dataloaders(cfg) -> Tuple[DataLoader, DataLoader]:
    if cfg.dataset.lower() == "koniq":
        train_ds = KonIQDataset(cfg.image_root, cfg.train_split)
        val_ds = KonIQDataset(cfg.image_root, cfg.val_split)
    elif cfg.dataset.lower() == "spaq":
        train_ds = SPAQDataset(cfg.image_root, cfg.train_split)
        val_ds = SPAQDataset(cfg.image_root, cfg.val_split)
    elif cfg.dataset.lower() == "kadid":
        train_ds = KADIDDataset(cfg.image_root, cfg.train_split)
        val_ds = KADIDDataset(cfg.image_root, cfg.val_split)
    elif cfg.dataset.lower() == "agiqa":
        train_ds = AGIQADataset(cfg.image_root, cfg.train_split)
        val_ds = AGIQADataset(cfg.image_root, cfg.val_split)
    else:
        raise ValueError(f"Unknown dataset {cfg.dataset}")
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)
    return train_loader, val_loader


def build_model(mode: str) -> nn.Module:
    mode = mode.lower()
    if mode == "resnet_baseline":
        return ResNetBaselineIQA()
    if mode == "stair_iqa":
        return StairIQA()
    if mode == "hyper_iqa":
        return HyperIQA()
    if mode == "caqf":
        return CAQF_IQA()
    if mode == "caqf_no_attn":
        return CAQF_IQA_NoAttn()
    if mode in ("tiny_r18", "tiny_r18_kd"):
        return TinyIQA_R18()
    raise ValueError(f"Unknown mode {mode}")


def train_stage2(config_path: str) -> None:
    raw = load_yaml_config(config_path)
    cfg = build_train_config(raw)
    set_global_seed(cfg.seed)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    train_loader, val_loader = build_dataloaders(cfg)
    model = build_model(cfg.mode).to(device)
    loss_fn = build_loss(cfg.loss)
    if cfg.mode == "tiny_r18_kd" and cfg.teacher_ckpt:
        teacher = ResNetBaselineIQA().to(device)
        load_model_ckpt(teacher, cfg.teacher_ckpt, strict=True)
        teacher.eval()
    else:
        teacher = None
    if cfg.optimizer.lower() == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    history: List[dict] = []
    best_metric = -1.0
    best_path = os.path.join(cfg.output_dir, f"{cfg.mode}_best.pth")
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        losses = []
        pbar = tqdm(train_loader, desc=f"Train Epoch {epoch}", leave=False)
        for imgs, mos in pbar:
            imgs = imgs.to(device)
            mos = mos.to(device)
            pred = model(imgs)
            if teacher is not None:
                with torch.no_grad():
                    t_pred = teacher(imgs)
                alpha = float(cfg.loss.get("kd_alpha", 0.5))
                kd = torch.mean((pred - t_pred) ** 2)
                mse = torch.mean((pred - mos) ** 2)
                loss = mse + alpha * kd
            else:
                loss = loss_fn(pred, mos)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{optimizer.param_groups[0]['lr']:.1e}")
        avg_train = float(sum(losses) / max(1, len(losses)))
        model.eval()
        val_losses = []
        all_y = []
        all_p = []
        with torch.no_grad():
            for imgs, mos in tqdm(val_loader, desc=f"Val Epoch {epoch}", leave=False):
                imgs = imgs.to(device)
                mos = mos.to(device)
                pred = model(imgs)
                val_losses.append(torch.mean((pred - mos) ** 2).item())
                all_y.extend(mos.view(-1).cpu().numpy().tolist())
                all_p.extend(pred.view(-1).cpu().numpy().tolist())
        val_plcc = plcc(all_y, all_p)
        val_srcc = srcc(all_y, all_p)
        val_loss = float(sum(val_losses) / max(1, len(val_losses)))
        rec = {"epoch": epoch, "train_loss": avg_train, "val_loss": val_loss, "val_plcc": val_plcc, "val_srcc": val_srcc}
        history.append(rec)
        save_history(history, os.path.join(cfg.output_dir, f"{cfg.mode}_history.json"))
        plot_training_curves(history, cfg.output_dir, title_prefix=cfg.mode)
        metric_for_best = val_srcc
        if metric_for_best > best_metric:
            best_metric = metric_for_best
            save_ckpt(model, optimizer, epoch, best_path)
    with open(os.path.join(cfg.output_dir, f"{cfg.mode}_best_metric.json"), "w", encoding="utf-8") as f:
        json.dump({"best_srcc": best_metric, "best_ckpt": best_path}, f, indent=2)

