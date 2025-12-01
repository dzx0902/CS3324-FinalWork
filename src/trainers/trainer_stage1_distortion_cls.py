import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from ..datasets.koniq_dataset import KonIQDistortionClsDataset
from ..models.backbones.resnet_backbone import ResNetBackbone50
from ..utils.logging_utils import ensure_dir, SimpleCSVLogger


class DistortionClassifier(nn.Module):
    def __init__(self, backbone: ResNetBackbone50, num_classes: int):
        super().__init__()
        self.backbone = backbone
        in_dim = backbone.feat_dims["f4"]
        self.cls_head = nn.Linear(in_dim, num_classes)

    def forward(self, x):
        _, _, f4 = self.backbone(x)
        z = torch.nn.functional.adaptive_avg_pool2d(f4, 1).flatten(1)
        logits = self.cls_head(z)
        return logits


def train_stage1(config: dict) -> str:
    device = torch.device("cuda" if torch.cuda.is_available() and config.get("device", "cuda") == "cuda" else "cpu")
    save_dir = config.get("save_dir", "checkpoints")
    ensure_dir(save_dir)
    dataset_root = config["dataset_root"]
    csv_path = config.get("csv_path")
    batch_size = int(config.get("batch_size", 32))
    epochs = int(config.get("epochs", 20))
    lr = float(config.get("learning_rate", 1e-3))
    num_classes = int(config.get("num_classes", 10))
    image_size = int(config.get("image_size", 224))
    freeze_stem = bool(config.get("freeze_stem_layer1", True))

    ds = KonIQDistortionClsDataset(dataset_root, csv_path=csv_path, image_size=image_size)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

    backbone = ResNetBackbone50(pretrained=True)
    if freeze_stem:
        for p in backbone.stem.parameters():
            p.requires_grad = False
        for p in backbone.layer1.parameters():
            p.requires_grad = False
    model = DistortionClassifier(backbone, num_classes).to(device)

    opt = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    ce = nn.CrossEntropyLoss()

    logger = SimpleCSVLogger(os.path.join(save_dir, "stage1_log.csv"), ["epoch", "loss", "acc"])
    best_acc = 0.0
    best_path = os.path.join(save_dir, "stage1_best.pth")

    for ep in range(1, epochs + 1):
        model.train()
        total = 0
        correct = 0
        running_loss = 0.0
        for x, y in dl:
            x = x.to(device)
            y = y.to(device)
            opt.zero_grad()
            logits = model(x)
            loss = ce(logits, y)
            loss.backward()
            opt.step()
            running_loss += loss.item() * x.size(0)
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += x.size(0)
        epoch_loss = running_loss / max(total, 1)
        acc = correct / max(total, 1)
        logger.append({"epoch": ep, "loss": epoch_loss, "acc": acc})
        if acc > best_acc:
            best_acc = acc
            torch.save({"model_state_dict": model.state_dict()}, best_path)

    return best_path

