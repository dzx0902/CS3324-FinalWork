import argparse
from src.utils.config_utils import load_yaml_config
from src.trainers.trainer_stage2_mos_regression import train_stage2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    args = ap.parse_args()
    cfg = load_yaml_config(args.config)
    ckpt = train_stage2(cfg)
    print("stage2_best", ckpt)


if __name__ == "__main__":
    main()
