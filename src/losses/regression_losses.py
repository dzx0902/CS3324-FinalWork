import torch
import torch.nn.functional as F


def mse_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(pred.view(-1), target.view(-1))


def mae_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.l1_loss(pred.view(-1), target.view(-1))


def _rankdata(x: torch.Tensor) -> torch.Tensor:
    vals, idx = torch.sort(x.view(-1))
    ranks = torch.zeros_like(idx, dtype=torch.float32)
    ranks[idx] = torch.arange(1, x.numel() + 1, dtype=torch.float32, device=x.device)
    return ranks


def srcc_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    rp = _rankdata(pred.float())
    rt = _rankdata(target.float())
    pm = torch.mean(rp)
    tm = torch.mean(rt)
    cov = torch.mean((rp - pm) * (rt - tm))
    ps = torch.std(rp) + 1e-8
    ts = torch.std(rt) + 1e-8
    srcc = cov / (ps * ts)
    return 1.0 - srcc


def pairwise_rank_loss(pred: torch.Tensor, target: torch.Tensor, margin: float = 0.0) -> torch.Tensor:
    p = pred.view(-1)
    t = target.view(-1)
    diff_t = t[:, None] - t[None, :]
    diff_p = p[:, None] - p[None, :]
    sign = torch.sign(diff_t)
    loss = F.relu(margin - sign * diff_p)
    mask = (diff_t != 0).float()
    loss = (loss * mask).sum() / (mask.sum() + 1e-8)
    return loss

