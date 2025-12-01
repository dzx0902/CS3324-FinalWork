import torch


def compute_plcc(y_pred: torch.Tensor, y_gt: torch.Tensor) -> float:
    y_pred = y_pred.flatten().float()
    y_gt = y_gt.flatten().float()
    pred_mean = torch.mean(y_pred)
    gt_mean = torch.mean(y_gt)
    cov = torch.mean((y_pred - pred_mean) * (y_gt - gt_mean))
    pred_std = torch.std(y_pred) + 1e-8
    gt_std = torch.std(y_gt) + 1e-8
    plcc = (cov / (pred_std * gt_std)).item()
    return float(plcc)


def _rankdata(x: torch.Tensor) -> torch.Tensor:
    vals, idx = torch.sort(x)
    ranks = torch.zeros_like(idx, dtype=torch.float32)
    ranks[idx] = torch.arange(1, x.numel() + 1, dtype=torch.float32, device=x.device)
    return ranks


def compute_srcc(y_pred: torch.Tensor, y_gt: torch.Tensor) -> float:
    rp = _rankdata(y_pred.flatten().float())
    rg = _rankdata(y_gt.flatten().float())
    return compute_plcc(rp, rg)

