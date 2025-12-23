import argparse
import os
import sys
import yaml
import json
from src.trainers.trainer_stage1_pretrain import pretrain
from src.trainers.trainer_stage2_regression import train_stage2
from src.trainers.trainer_eval_only import build_loader, eval_model, plot_pred_vs_mos
from src.models.iqa_heads.resnet_baseline import ResNetBaselineIQA
from src.models.iqa_heads.stair_iqa import StairIQA
from src.models.iqa_heads.hyper_iqa import HyperIQA
from src.models.iqa_heads.caqf_iqa import CAQF_IQA, CAQF_IQA_NoAttn
from src.models.iqa_heads.tiny_iqa import TinyIQA_R18, TinyIQA_R34, TinyIQA_R18_MS
from src.utils.checkpoint_utils import load_model_ckpt
import torch


CONFIG_ALIASES = {
    "baseline": "configs/train_koniq_baseline.yaml",
    "stair": "configs/train_koniq_stair.yaml",
    "hyper": "configs/train_koniq_hyper.yaml",
    "caqf": "configs/train_koniq_caqf.yaml",
    "caqf_no_attn": "configs/train_koniq_caqf_no_attn.yaml",
    "tiny_r18": "configs/train_koniq_tiny.yaml",
    "tiny_r18_kd_stageA": "configs/train_koniq_tiny_r18_kd_stageA.yaml",
    "tiny_r18_kd_stageB": "configs/train_koniq_tiny_r18_kd_stageB.yaml",
    "tiny_r34": "configs/train_koniq_tiny_r34.yaml",
    "tiny_r34_kd_stageA": "configs/train_koniq_tiny_r34_kd_stageA.yaml",
    "tiny_r34_kd_stageB": "configs/train_koniq_tiny_r34_kd_stageB.yaml",
    "tiny_r18_ms": "configs/train_koniq_tiny_r18_ms.yaml",
    "tiny_r18_ms_kd_stageA": "configs/train_koniq_tiny_r18_ms_kd_stageA.yaml",
    "tiny_r18_ms_kd_stageB": "configs/train_koniq_tiny_r18_ms_kd_stageB.yaml",
    "tiny_r34_mse": "configs/train_koniq_tiny_r34_mse.yaml",
    "tiny_r34_mse_srcc": "configs/train_koniq_tiny_r34_mse_srcc.yaml",
}
EVAL_DEFAULT = "configs/eval_cross_dataset.yaml"
SUITE_DEFAULT = "configs/ablation_suite.yaml"


def build_model(name: str):
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
        return TinyIQA_R18(hidden_dim1=256, hidden_dim2=0)
    if n == "tiny_r34":
        return TinyIQA_R34(use_se=True, hidden_dim1=512, hidden_dim2=256)
    if n == "tiny_r18_ms":
        return TinyIQA_R18_MS()
    if n == "tiny_r18_kd":
        return TinyIQA_R18(hidden_dim1=256, hidden_dim2=0)
    if n == "tiny_r34_kd":
        return TinyIQA_R34(use_se=True, hidden_dim1=512, hidden_dim2=256)
    if n == "tiny_r18_ms_kd":
        return TinyIQA_R18_MS()
    return TinyIQA_R18()


def cmd_stage1(args):
    pretrain(args.image_root, args.file_list, args.out_ckpt, args.epochs, args.batch_size, args.device)


def cmd_train(args):
    cfg = args.config
    if args.alias:
        cfg = CONFIG_ALIASES.get(args.alias, cfg)
    if not cfg:
        raise SystemExit("Missing --config or --alias")
    train_stage2(cfg)


def cmd_eval(args):
    cfg_path = args.config or EVAL_DEFAULT
    with open(cfg_path, "r", encoding="utf-8") as f:
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
            loader = build_loader(d, image_roots[d], metas[d], batch_size=args.batch_size, num_workers=args.num_workers)
            pl, sr, rm, ma, y, p = eval_model(model, loader, device)
            model_results[d] = {"plcc": pl, "srcc": sr, "rmse": rm, "mae": ma}
            scatter_path = os.path.join(out_dir, f"{m}_{d}_scatter.png")
            plot_pred_vs_mos(y, p, scatter_path, title=f"{m} on {d}")
        results[m] = model_results
    with open(os.path.join(out_dir, "eval_results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)


def cmd_ablation(args):
    suite_path = args.suite or SUITE_DEFAULT
    with open(suite_path, "r", encoding="utf-8") as f:
        suite = yaml.safe_load(f)
    for cfg_path in suite.get("train_configs", []):
        train_stage2(cfg_path)
    eval_cfg = suite.get("eval_config") or EVAL_DEFAULT
    class _A: pass
    a = _A()
    a.config = eval_cfg
    a.batch_size = 64
    a.num_workers = 4
    cmd_eval(a)


def cmd_tables(args):
    from make_tables import to_markdown, to_latex
    with open(args.results, "r", encoding="utf-8") as f:
        results = json.load(f)
    os.makedirs(os.path.dirname(args.out_md), exist_ok=True)
    to_markdown(results, args.out_md)
    to_latex(results, args.out_tex)


def cmd_stats(args):
    from scripts.collect_model_stats import run as run_stats
    run_stats()


def cmd_pipeline(args):
    train_stage2(CONFIG_ALIASES["baseline"])
    train_stage2(CONFIG_ALIASES["tiny_r18_kd_stageA"])
    train_stage2(CONFIG_ALIASES["tiny_r18_kd_stageB"])
    train_stage2(CONFIG_ALIASES["tiny_r34"])
    train_stage2(CONFIG_ALIASES["tiny_r34_kd_stageA"])
    train_stage2(CONFIG_ALIASES["tiny_r34_kd_stageB"])
    eval_cfg = EVAL_DEFAULT
    class _A: pass
    a = _A()
    a.config = eval_cfg
    a.batch_size = 64
    a.num_workers = 4
    cmd_eval(a)
    class _B: pass
    b = _B()
    b.results = "results/eval_results.json"
    b.out_md = "results/table.md"
    b.out_tex = "results/table.tex"
    cmd_tables(b)
    cmd_stats(None)


def cmd_delta(args):
    targets = [
        CONFIG_ALIASES["tiny_r18_kd_stageA"],
        CONFIG_ALIASES["tiny_r18_kd_stageB"],
        CONFIG_ALIASES["tiny_r34"],
        CONFIG_ALIASES["tiny_r34_mse"],
        CONFIG_ALIASES["tiny_r34_mse_srcc"],
        CONFIG_ALIASES["tiny_r18_ms"],
        CONFIG_ALIASES["tiny_r18_ms_kd_stageA"],
        CONFIG_ALIASES["tiny_r18_ms_kd_stageB"],
    ]
    for cfg in targets:
        train_stage2(cfg)
    class _A: pass
    a = _A()
    a.config = EVAL_DEFAULT
    a.batch_size = 64
    a.num_workers = 4
    cmd_eval(a)
    class _B: pass
    b = _B()
    b.results = "results/eval_results.json"
    b.out_md = "results/table.md"
    b.out_tex = "results/table.tex"
    cmd_tables(b)
    cmd_stats(None)

def cmd_delta_eval(args):
    class _A: pass
    a = _A()
    a.config = EVAL_DEFAULT
    a.batch_size = 64
    a.num_workers = 4
    cmd_eval(a)
    class _B: pass
    b = _B()
    b.results = "results/eval_results.json"
    b.out_md = "results/table.md"
    b.out_tex = "results/table.tex"
    cmd_tables(b)
    cmd_stats(None)

def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers()
    s1 = sub.add_parser("stage1")
    s1.add_argument("--image_root", type=str, required=True)
    s1.add_argument("--file_list", type=str, required=True)
    s1.add_argument("--out_ckpt", type=str, default="outputs/stage1_pretrain.pth")
    s1.add_argument("--epochs", type=int, default=1)
    s1.add_argument("--batch_size", type=int, default=16)
    s1.add_argument("--device", type=str, default="cuda")
    s1.set_defaults(func=cmd_stage1)
    tr = sub.add_parser("train")
    tr.add_argument("--config", type=str)
    tr.add_argument("--alias", type=str, choices=list(CONFIG_ALIASES.keys()))
    tr.set_defaults(func=cmd_train)
    ev = sub.add_parser("eval")
    ev.add_argument("--config", type=str, default=EVAL_DEFAULT)
    ev.add_argument("--batch_size", type=int, default=64)
    ev.add_argument("--num_workers", type=int, default=4)
    ev.set_defaults(func=cmd_eval)
    ab = sub.add_parser("ablation")
    ab.add_argument("--suite", type=str, default=SUITE_DEFAULT)
    ab.set_defaults(func=cmd_ablation)
    tb = sub.add_parser("tables")
    tb.add_argument("--results", type=str, default="results/eval_results.json")
    tb.add_argument("--out_md", type=str, default="results/table.md")
    tb.add_argument("--out_tex", type=str, default="results/table.tex")
    tb.set_defaults(func=cmd_tables)
    st = sub.add_parser("stats")
    st.set_defaults(func=cmd_stats)
    pl = sub.add_parser("pipeline")
    pl.set_defaults(func=cmd_pipeline)
    dl = sub.add_parser("delta")
    dl.set_defaults(func=cmd_delta)
    de = sub.add_parser("delta_eval")
    de.set_defaults(func=cmd_delta_eval)
    args = p.parse_args()
    if hasattr(args, "func"):
        args.func(args)
    else:
        # 默认执行整套流水线：baseline → Tiny KD A/B → R34 → 评估 → 表格 → 统计
        cmd_pipeline(None)


if __name__ == "__main__":
    main()
