import argparse
import os
from src.trainers.trainer_stage2_regression import train_stage2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    train_stage2(args.config)


if __name__ == "__main__":
    main()

