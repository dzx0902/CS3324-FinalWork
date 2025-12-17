import argparse
from src.trainers.trainer_stage1_pretrain import pretrain


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_root", type=str, required=True)
    parser.add_argument("--file_list", type=str, required=True)
    parser.add_argument("--out_ckpt", type=str, default="outputs/stage1_pretrain.pth")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    pretrain(args.image_root, args.file_list, args.out_ckpt, args.epochs, args.batch_size, args.device)


if __name__ == "__main__":
    main()

