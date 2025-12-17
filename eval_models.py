import argparse
import os
import torch
import yaml
from src.models.iqa_heads.resnet_baseline import ResNetBaselineIQA
from src.models.iqa_heads.stair_iqa import StairIQA
from src.models.iqa_heads.hyper_iqa import HyperIQA
from src.models.iqa_heads.caqf_iqa import CAQF_IQA, CAQF_IQA_NoAttn
from src.models.iqa_heads.tiny_iqa import TinyIQA_R18
from src.utils.checkpoint_utils import load_model_ckpt
from src.trainers.trainer_eval_only import build_loader, eval_model, plot_pred_vs_mos


def build_model(name: str) -> torch.nn.Module:
    n = name.lower()
    if n == "resnet_baseline":
        return ResNetBaselineIQA()
    if n == "stair_iqa":
        return StairIQA()
    if n == "hyper_iqa":
        return HyperIQA()
    if n == "caqf":
        return CAQF_IQA()
    if n == "caqf_no_attn":
        return CAQF_IQA_NoAttn()
    if n == "tiny_r18":
        return TinyIQA_R18()
    if n == "tiny_r18_kd":
        return TinyIQA_R18()
    raise ValueError(f"Unknown model {name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    models = cfg["models"]
    datasets = cfg["datasets"]
    image_roots = cfg["image_roots"]
    metas = cfg["metas"]
    ckpts = cfg["ckpts"]
    out_dir = cfg.get("out_dir", "results")
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = {}
    for m in models:
        model = build_model(m).to(device)
        load_model_ckpt(model, ckpts[m], strict=False)
        model_results = {}
        for d in datasets:
            loader = build_loader(d, image_roots[d], metas[d], batch_size=64, num_workers=4)
            pl, sr, rm, ma, y, p = eval_model(model, loader, device)
            model_results[d] = {"plcc": pl, "srcc": sr, "rmse": rm, "mae": ma}
            scatter_path = os.path.join(out_dir, f"{m}_{d}_scatter.png")
            plot_pred_vs_mos(y, p, scatter_path, title=f"{m} on {d}")
        results[m] = model_results
    with open(os.path.join(out_dir, "eval_results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()

