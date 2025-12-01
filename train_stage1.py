import argparse
from src.utils.config_utils import load_yaml_config
from src.trainers.trainer_stage1_distortion_cls import train_stage1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    args = ap.parse_args()
    cfg = load_yaml_config(args.config)
    ckpt = train_stage1(cfg)
    print("stage1_best", ckpt)


if __name__ == "__main__":
    main()
