import argparse
import os
import yaml
from src.trainers.trainer_stage2_regression import train_stage2
from eval_models import main as eval_main


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", type=str, required=True)
    args = parser.parse_args()
    with open(args.suite, "r", encoding="utf-8") as f:
        suite = yaml.safe_load(f)
    for cfg_path in suite.get("train_configs", []):
        train_stage2(cfg_path)
    eval_cfg = suite.get("eval_config")
    if eval_cfg:
        import sys
        sys.argv = ["eval_models.py", "--config", eval_cfg]
        eval_main()


if __name__ == "__main__":
    main()

