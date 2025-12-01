import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from ..datasets.koniq_dataset import KonIQMOSDataset
from ..models.backbones.resnet_backbone import ResNetBackbone50
from ..models.iqa_heads.caqf_iqa import CAQF_IQA
from ..models.iqa_heads.resnet_baseline import ResNetBaselineIQA
from ..losses.regression_losses import mse_loss, srcc_loss
from ..utils.metrics import compute_plcc, compute_srcc
from ..utils.logging_utils import ensure_dir, SimpleCSVLogger


def _load_backbone_from_stage1(backbone: ResNetBackbone50, ckpt_path: str):
    if not ckpt_path or not os.path.isfile(ckpt_path):
        return
    sd = torch.load(ckpt_path, map_location="cpu")["model_state_dict"]
    bk = {}
    for k, v in sd.items():
        if k.startswith("backbone."):
            bk[k[len("backbone."):]] = v
    backbone.load_state_dict(bk, strict=False)


def train_stage2(config: dict) -> str:
    device = torch.device("cuda" if torch.cuda.is_available() and config.get("device", "cuda") == "cuda" else "cpu")
    save_dir = config.get("save_dir", "checkpoints")
    ensure_dir(save_dir)
    dataset_root = config["dataset_root"]
    csv_path = config.get("mos_csv_path") or config.get("csv_path")
    batch_size = int(config.get("batch_size", 32))
    epochs = int(config.get("epochs", 30))
    lr = float(config.get("learning_rate", 5e-4))
    image_size = int(config.get("image_size", 224))
    mode = config.get("mode", "ours_stage1_pretrained")
    lambda_srcc = float(config.get("lambda_srcc", 0.0))
    stage1_ckpt = config.get("stage1_ckpt")
    train_list_json = config.get("train_list_json")
    test_list_json = config.get("test_list_json")

    ds_tr = KonIQMOSDataset(dataset_root, csv_path, split="train", image_size=image_size, file_list_json=train_list_json)
    ds_va = KonIQMOSDataset(dataset_root, csv_path, split="val", image_size=image_size, file_list_json=train_list_json)
    dl_tr = DataLoader(ds_tr, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    dl_va = DataLoader(ds_va, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    backbone = ResNetBackbone50(pretrained=True)
    if mode == "resnet_baseline":
        model = ResNetBaselineIQA(backbone).to(device)
    else:
        model = CAQF_IQA(backbone).to(device)
    if mode == "ours_stage1_pretrained" and stage1_ckpt:
        _load_backbone_from_stage1(backbone, stage1_ckpt)

    opt = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    logger = SimpleCSVLogger(os.path.join(save_dir, "stage2_log.csv"), ["epoch", "loss", "val_plcc", "val_srcc"])
    best_metric = -1e9
    best_path = os.path.join(save_dir, f"stage2_best_{mode}.pth")

    for ep in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        count = 0
        for x, y in dl_tr:
            x = x.to(device)
            y = y.to(device)
            opt.zero_grad()
            pred = model(x)
            loss = mse_loss(pred, y)
            if lambda_srcc > 0.0:
                loss = loss + lambda_srcc * srcc_loss(pred, y)
            loss.backward()
            opt.step()
            running_loss += loss.item() * x.size(0)
            count += x.size(0)
        tr_loss = running_loss / max(count, 1)

        model.eval()
        preds = []
        gts = []
        with torch.no_grad():
            for x, y in dl_va:
                x = x.to(device)
                pred = model(x)
                preds.append(pred.cpu())
                gts.append(y)
        preds = torch.cat(preds)
        gts = torch.cat(gts)
        plcc = compute_plcc(preds, gts)
        srcc = compute_srcc(preds, gts)
        logger.append({"epoch": ep, "loss": tr_loss, "val_plcc": plcc, "val_srcc": srcc})
        metric = srcc
        if metric > best_metric:
            best_metric = metric
            torch.save({"model_state_dict": model.state_dict()}, best_path)

    return best_path

