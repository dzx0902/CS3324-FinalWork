"""Evaluation metrics for NR-IQA: PLCC, SRCC, RMSE, MAE."""
from __future__ import annotations
import numpy as np


def plcc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    y_true = y_true - y_true.mean()
    y_pred = y_pred - y_pred.mean()
    denom = np.sqrt((y_true ** 2).sum()) * np.sqrt((y_pred ** 2).sum())
    if denom == 0:
        return 0.0
    return float((y_true * y_pred).sum() / denom)


def srcc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    # Rank with average for ties
    def rankdata(a: np.ndarray) -> np.ndarray:
        temp = a.argsort()
        ranks = np.empty_like(temp, dtype=float)
        ranks[temp] = np.arange(len(a), dtype=float)
        # handle ties
        _, inv, counts = np.unique(a, return_inverse=True, return_counts=True)
        cumulative = np.cumsum(counts)
        starts = cumulative - counts
        avg_ranks = (starts + cumulative - 1) / 2.0
        ranks = avg_ranks[inv]
        return ranks
    rx = rankdata(y_true)
    ry = rankdata(y_pred)
    return plcc(rx, ry)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))

