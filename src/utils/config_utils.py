"""Utility functions for loading and validating YAML experiment configs."""
from __future__ import annotations
import os
from dataclasses import dataclass
from typing import Any, Dict
import yaml


@dataclass
class TrainConfig:
    mode: str
    dataset: str
    train_split: str
    val_split: str
    image_root: str
    output_dir: str
    epochs: int = 20
    batch_size: int = 32
    lr: float = 1e-4
    weight_decay: float = 0.0
    optimizer: str = "adamw"
    loss: Dict[str, Any] = None
    teacher_ckpt: str | None = None
    stage1_ckpt: str | None = None
    num_workers: int = 4
    device: str = "cuda"
    seed: int = 42


def load_yaml_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


def build_train_config(cfg: Dict[str, Any]) -> TrainConfig:
    out = TrainConfig(
        mode=cfg["mode"],
        dataset=cfg["dataset"],
        train_split=cfg["train_split"],
        val_split=cfg["val_split"],
        image_root=cfg["image_root"],
        output_dir=cfg.get("output_dir", "outputs"),
        epochs=int(cfg.get("epochs", 20)),
        batch_size=int(cfg.get("batch_size", 32)),
        lr=float(cfg.get("lr", 1e-4)),
        weight_decay=float(cfg.get("weight_decay", 0.0)),
        optimizer=str(cfg.get("optimizer", "adamw")),
        loss=cfg.get("loss", {"type": "mse"}),
        teacher_ckpt=cfg.get("teacher_ckpt"),
        stage1_ckpt=cfg.get("stage1_ckpt"),
        num_workers=int(cfg.get("num_workers", 4)),
        device=str(cfg.get("device", "cuda")),
        seed=int(cfg.get("seed", 42)),
    )
    os.makedirs(out.output_dir, exist_ok=True)
    return out

