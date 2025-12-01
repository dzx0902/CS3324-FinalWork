import torch
from torch.utils.data import DataLoader
from ..utils.metrics import compute_plcc, compute_srcc


def evaluate_model_on_dataset(model, dataset, batch_size: int, device: torch.device):
    dl = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    model.eval()
    preds = []
    gts = []
    with torch.no_grad():
        for x, y in dl:
            x = x.to(device)
            p = model(x)
            preds.append(p.cpu())
            gts.append(y)
    preds = torch.cat(preds)
    gts = torch.cat(gts)
    return {
        "plcc": compute_plcc(preds, gts),
        "srcc": compute_srcc(preds, gts),
    }

