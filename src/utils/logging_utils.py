"""History saving and matplotlib plotting utilities for training/evaluation."""
from __future__ import annotations
import json
import os
from typing import List, Dict
import matplotlib.pyplot as plt


def save_history(history: List[Dict], out_path: str) -> None:
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)


def plot_training_curves(history: List[Dict], out_dir: str, title_prefix: str = "") -> None:
    epochs = [h["epoch"] for h in history]
    train_loss = [h["train_loss"] for h in history]
    val_loss = [h["val_loss"] for h in history]
    val_plcc = [h.get("val_plcc", 0.0) for h in history]
    val_srcc = [h.get("val_srcc", 0.0) for h in history]

    plt.figure()
    plt.plot(epochs, train_loss, label="train_loss")
    plt.plot(epochs, val_loss, label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{title_prefix} Loss Curves")
    plt.legend()
    plt.grid(True)
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(os.path.join(out_dir, f"{title_prefix}_loss_curve.png"))
    plt.close()

    plt.figure()
    plt.plot(epochs, val_plcc, label="val_PLCC")
    plt.plot(epochs, val_srcc, label="val_SRCC")
    plt.xlabel("Epoch")
    plt.ylabel("Correlation")
    plt.title(f"{title_prefix} PLCC/SRCC Curves")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(out_dir, f"{title_prefix}_corr_curve.png"))
    plt.close()

