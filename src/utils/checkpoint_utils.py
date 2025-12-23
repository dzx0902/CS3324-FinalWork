"""Checkpoint save/load helpers supporting teacher/student models."""
from __future__ import annotations
import os
from typing import Any, Dict
import torch


def save_ckpt(model: torch.nn.Module, optimizer: torch.optim.Optimizer, epoch: int, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch}, path)


def load_model_ckpt(model: torch.nn.Module, path: str, strict: bool = True) -> Dict[str, Any]:
    state = torch.load(path, map_location="cpu")
    target = state.get("model", {})
    current = model.state_dict()
    filtered = {}
    for k, v in target.items():
        if k in current and current[k].shape == v.shape:
            filtered[k] = v
    missing = set(current.keys()) - set(filtered.keys())
    unexpected = set(target.keys()) - set(filtered.keys())
    model.load_state_dict(filtered, strict=False)
    return {"model": filtered, "optimizer": state.get("optimizer"), "epoch": state.get("epoch"), "missing": list(missing), "unexpected": list(unexpected)}

