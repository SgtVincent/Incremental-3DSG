import numpy as np
from pathlib import Path
import torch
from torch_geometric.data import DataLoader
import torch_geometric.transforms as T
import torch.nn.functional as F

from models.model import parse_model
from clf_utils.dataset import ScanNetDataset
from clf_utils.scannet_data_utils import *
from clf_utils.config import parse_args


def train(model, optimizer, train_loader, device):
    model.train()
    total_loss = 0
    total_correct = 0

    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()  # Clear gradients.
        out = model(data.x, data.edge_index, data.batch)  # Forward pass.
        loss = F.nll_loss(out, data.y.view(-1))  # Loss computation.
        loss.backward()  # Backward pass.
        optimizer.step()  # Update model parameters.
        total_loss += loss.item() * data.num_graphs

        pred = out.argmax(dim=-1)
        total_correct += int((pred == data.y).sum())

    train_loss = total_loss / len(train_loader.dataset)
    train_acc = total_correct / len(train_loader.dataset)
    return train_loss, train_acc


@torch.no_grad()
def test(model, test_loader, device):
    model.eval()
    criterion = torch.nn.CrossEntropyLoss()
    total_loss = 0
    total_correct = 0

    for data in test_loader:
        data = data.to(device)
        out = model(data.x, data.edge_index, data.batch)
        loss = F.nll_loss(out, data.y.view(-1))

        total_loss += loss.item() * data.num_graphs
        pred = out.argmax(dim=-1)
        total_correct += int((pred == data.y).sum())

    test_loss = total_loss / len(test_loader.dataset)
    test_acc = total_correct / len(test_loader.dataset)
    return test_loss, test_acc

def main():

    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # ------------------- data preparation -------------------- #

    with open(args.train_split, 'r') as f:
        train_scenes = f.read().splitlines()
    with open(args.val_split, 'r') as f:
        val_scenes = f.read().splitlines()
    # debug dataset
    # with open(args.train_split, 'r') as f:
    #     train_scenes = ["scene0000_00", "scene0000_01", "scene0000_02"]
    # with open(args.val_split, 'r') as f:
    #     val_scenes = ["scene0001_00", "scene0001_01"]

    # pre_transform, transform = T.NormalizeScale(), T.SamplePoints(1024)
    pre_transform, transform = None, None
    train_dataset = ScanNetDataset(args.root, train_scenes, split="train",
                                   transform=transform, pre_transform=pre_transform)
    val_dataset = ScanNetDataset(args.root, val_scenes, split="val")

    train_loader = DataLoader(
        train_dataset, batch_size=1, shuffle=True, num_workers=4)
    val_loader = DataLoader(
        val_dataset, batch_size=1, shuffle=False, num_workers=4)

    # --------------- construct model, optimizer ------------------------- #
    model = parse_model(args.model)
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)  # Only perform weight-decay on first convolution.

    best_val_acc = 0
    for epoch in range(args.num_epoch):
        train_loss, train_acc = train(model, optimizer, train_loader, device)

        if epoch % args.epochs_per_val == 0:
            val_loss, val_acc = test(model, val_loader, device)
            log = 'Epoch: {:03d}, train_loss: {:.4f}, train_acc: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}'
            print(log.format(epoch, train_loss, train_acc, val_loss, val_acc))
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), args.save_model_path)
        else:
            log = 'Epoch: {:03d}, train_loss: {:.4f}, train_acc: {:.4f}'
            print(log.format(epoch, train_loss, train_acc))


if __name__ == '__main__':
    main()