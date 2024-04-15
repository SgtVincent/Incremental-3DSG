import argparse

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--root', type=str, default="D:/datasets/ScanNet")
    parser.add_argument('--train_split', type=str, default="./split/train.txt")
    parser.add_argument('--val_split', type=str, default="./split/val.txt")
    parser.add_argument('--model', type=str, default="GCN")
    parser.add_argument('--save_model_path', type=str, default="./data/best_gcn.pt")
    parser.add_argument('--epochs_per_val', type=int, default=10)
    parser.add_argument('--num_epoch', type=int, default=100)

    args = parser.parse_args()
    return args