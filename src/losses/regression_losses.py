"""Regression and correlation losses for NR-IQA."""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class MSELoss(nn.Module):
    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(y_pred, y_true)


class MAELoss(nn.Module):
    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        return F.l1_loss(y_pred, y_true)


class SRCCLoss(nn.Module):
    """
    Differentiable proxy for SRCC loss using Pearson Correlation.
    Since sorting is non-differentiable, we optimize for linear correlation (Pearson),
    which is a strong proxy for rank correlation (Spearman).
    """
    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        # Flatten and center
        vx = y_pred.view(-1) - torch.mean(y_pred)
        vy = y_true.view(-1) - torch.mean(y_true)
        # Compute Pearson correlation
        cost = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)) + 1e-8)
        return 1.0 - cost


class PairwiseRankLoss(nn.Module):
    """Enforce pairwise order consistency."""
    def __init__(self, margin: float = 0.0):
        super().__init__()
        self.margin = margin

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        n = y_true.shape[0]
        if n < 2:
            return torch.zeros((), device=y_true.device)
        diff_true = y_true.unsqueeze(0) - y_true.unsqueeze(1)
        diff_pred = y_pred.unsqueeze(0) - y_pred.unsqueeze(1)
        sign_true = torch.sign(diff_true)
        loss = F.relu(self.margin - sign_true * diff_pred)
        return loss.mean()


def build_loss(config: dict) -> nn.Module:
    t = config.get("type", "mse")
    if t == "mse":
        return MSELoss()
    if t == "mae":
        return MAELoss()
    if t == "mse_srcc":
        alpha = float(config.get("alpha", 1.0))
        class Combo(nn.Module):
            def __init__(self):
                super().__init__()
                self.mse = MSELoss()
                self.srcc = SRCCLoss()
            def forward(self, y_pred, y_true):
                return self.mse(y_pred, y_true) + alpha * self.srcc(y_pred, y_true)
        return Combo()
    if t == "mse_rank":
        alpha = float(config.get("alpha", 1.0))
        margin = float(config.get("margin", 0.0))
        class Combo(nn.Module):
            def __init__(self):
                super().__init__()
                self.mse = MSELoss()
                self.rank = PairwiseRankLoss(margin=margin)
            def forward(self, y_pred, y_true):
                return self.mse(y_pred, y_true) + alpha * self.rank(y_pred, y_true)
        return Combo()
    raise ValueError(f"Unknown loss type {t}")

