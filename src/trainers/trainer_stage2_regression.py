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
from ..models.iqa_heads.tiny_iqa import TinyIQA_R18, TinyIQA_R34, TinyIQA_R18_MS


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


def build_model_with_cfg(cfg) -> nn.Module:
    mode = str(cfg.mode).lower()
    margs = cfg.model or {}
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
        return TinyIQA_R18(
            dropout=float(margs.get("dropout_p", 0.0)),
            mos_min=float(margs.get("mos_min", 0.0)),
            mos_max=float(margs.get("mos_max", 100.0)),
            use_se=bool(margs.get("use_se", False)),
            hidden_dim1=int(margs.get("hidden_dim1", 256)),
            hidden_dim2=int(margs.get("hidden_dim2", 0)),
        )
    if mode in ("tiny_r34", "tiny_r34_kd"):
        return TinyIQA_R34(
            dropout=float(margs.get("dropout_p", 0.1)),
            mos_min=float(margs.get("mos_min", 0.0)),
            mos_max=float(margs.get("mos_max", 100.0)),
            use_se=bool(margs.get("use_se", False)),
            hidden_dim1=int(margs.get("hidden_dim1", 512)),
            hidden_dim2=int(margs.get("hidden_dim2", 0)),
        )
    if mode in ("tiny_r18_ms", "tiny_r18_ms_kd"):
        return TinyIQA_R18_MS(
            C=int(margs.get("C", 256)),
            dropout=float(margs.get("dropout_p", 0.1)),
            mos_min=float(margs.get("mos_min", 0.0)),
            mos_max=float(margs.get("mos_max", 100.0)),
        )
    raise ValueError(f"Unknown mode {mode}")


def build_model_and_teacher(cfg, device):
    m = build_model_with_cfg(cfg).to(device)
    t = None
    kd_cfg = cfg.kd or {}
    if str(cfg.mode).lower().endswith("_kd") or kd_cfg.get("enabled", False):
        t = ResNetBaselineIQA().to(device)
        ck = kd_cfg.get("teacher_ckpt") or cfg.teacher_ckpt
        if ck:
            load_model_ckpt(t, ck, strict=False)
        t.eval()
        for p in t.parameters():
            p.requires_grad = False
    return m, t


def train_stage2(config_path: str) -> None:
    raw = load_yaml_config(config_path)
    cfg = build_train_config(raw)
    set_global_seed(cfg.seed)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    train_loader, val_loader = build_dataloaders(cfg)
    model, teacher = build_model_and_teacher(cfg, device)
    loss_fn = build_loss(cfg.loss)
    if cfg.optimizer.lower() == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    history: List[dict] = []
    best_metric = -1.0
    best_path = os.path.join(cfg.output_dir, f"{cfg.mode}_best.pth")
    kd_cfg = cfg.kd or {}
    kd_stage = str(kd_cfg.get("stage", "")).upper()
    if kd_stage == "B" and kd_cfg.get("stageA_ckpt"):
        try:
            state = torch.load(kd_cfg["stageA_ckpt"], map_location="cpu")
            model.load_state_dict(state["model"], strict=False)
        except Exception:
            pass
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        losses = []
        pbar = tqdm(train_loader, desc=f"Train Epoch {epoch} [{cfg.mode} {kd_stage or 'S'}]", leave=False)
        for imgs, mos in pbar:
            imgs = imgs.to(device)
            mos = mos.to(device)
            pred = model(imgs)
            if teacher is not None:
                with torch.no_grad():
                    t_pred = teacher(imgs)
                if kd_stage == "A":
                    loss = torch.mean((pred - t_pred) ** 2)
                else:
                    half = int(kd_cfg.get("kd_half_epochs", 0))
                    use_kd_now = True
                    if half > 0:
                        use_kd_now = epoch <= min(half, cfg.epochs)
                    elif cfg.epochs >= 2:
                        use_kd_now = epoch <= (cfg.epochs // 2)
                    alpha = float(kd_cfg.get("alpha", cfg.loss.get("kd_alpha", 0.1)))
                    kd = torch.mean((pred - t_pred) ** 2)
                    mse = torch.mean((pred - mos) ** 2)
                    loss = mse + (alpha * kd if use_kd_now else 0.0)
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
        rec = {"epoch": epoch, "train_loss": avg_train, "val_loss": val_loss, "val_plcc": val_plcc, "val_srcc": val_srcc, "stage": kd_stage or "S"}
        history.append(rec)
        save_history(history, os.path.join(cfg.output_dir, f"{cfg.mode}_history.json"))
        plot_training_curves(history, cfg.output_dir, title_prefix=f"{cfg.mode}_{kd_stage or 'S'}")
        metric_for_best = val_srcc
        if metric_for_best > best_metric:
            best_metric = metric_for_best
            save_ckpt(model, optimizer, epoch, best_path)
    with open(os.path.join(cfg.output_dir, f"{cfg.mode}_best_metric.json"), "w", encoding="utf-8") as f:
        json.dump({"best_srcc": best_metric, "best_ckpt": best_path}, f, indent=2)

