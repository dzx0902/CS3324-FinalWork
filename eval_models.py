import argparse
import json
import os
import torch

from src.utils.config_utils import load_yaml_config
from src.utils.logging_utils import ensure_dir
from src.datasets.koniq_dataset import KonIQMOSDataset
from src.models.backbones.resnet_backbone import ResNetBackbone50
from src.models.iqa_heads.caqf_iqa import CAQF_IQA
from src.models.iqa_heads.resnet_baseline import ResNetBaselineIQA
from src.trainers.trainer_eval_only import evaluate_model_on_dataset


def load_model(name: str, ckpt: str, device: torch.device):
    backbone = ResNetBackbone50(pretrained=False)
    if name == "resnet_baseline":
        model = ResNetBaselineIQA(backbone)
    else:
        model = CAQF_IQA(backbone)
    sd = torch.load(ckpt, map_location="cpu").get("model_state_dict", None)
    if sd:
        model.load_state_dict(sd, strict=False)
    return model.to(device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/eval_cross_dataset.yaml")
    ap.add_argument("--csv", type=str, default="Data/metas/koniq10k_scores_and_distributions.csv")
    ap.add_argument("--root", type=str, default="Data/koniq_test")
    ap.add_argument("--list_json", type=str, default="Data/metas/koniq_test.json")
    args = ap.parse_args()
    cfg = load_yaml_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ensure_dir("results")

    ds = KonIQMOSDataset(args.root, args.csv, split="test", image_size=cfg.get("image_size", 224), file_list_json=args.list_json)
    results = {}
    for name, ckpt in cfg["models"].items():
        if not os.path.isfile(ckpt):
            continue
        model = load_model(name, ckpt, device)
        metrics = evaluate_model_on_dataset(model, ds, batch_size=32, device=device)
        results[name] = metrics
    out_path = os.path.join("results", "eval_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(out_path)


if __name__ == "__main__":
    main()
