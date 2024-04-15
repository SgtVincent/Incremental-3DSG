from itertools import product

import argparse
from utils.dataset import ScanNetDataset
from benchmark.datasets import get_dataset
from benchmark.train_eval import train_eval

from benchmark.gcn import GCN, GCNWithJK
from benchmark.graph_sage import GraphSAGE, GraphSAGEWithJK
from benchmark.gin import GIN0, GIN0WithJK, GIN, GINWithJK
from benchmark.graclus import Graclus
from benchmark.top_k import TopK
from benchmark.sag_pool import SAGPool
from benchmark.diff_pool import DiffPool
from benchmark.edge_pool import EdgePool
from benchmark.global_attention import GlobalAttentionNet
from benchmark.set2set import Set2SetNet
from benchmark.sort_pool import SortPool
from benchmark.asap import ASAP

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--lr_decay_factor', type=float, default=0.5)
parser.add_argument('--lr_decay_step_size', type=int, default=50)
parser.add_argument('--root', type=str, default="D:/datasets/ScanNet")
parser.add_argument('--train_split', type=str, default="./split/train.txt")
parser.add_argument('--val_split', type=str, default="./split/val.txt")
args = parser.parse_args()

layers = [2, 3, 4, 5]
hiddens = [16, 32, 64]
# datasets = ['MUTAG', 'PROTEINS', 'IMDB-BINARY', 'REDDIT-BINARY']  # , 'COLLAB']
nets = [
    # GCNWithJK,
    # GraphSAGEWithJK,
    # GIN0WithJK,
    # GINWithJK,
    # Graclus,
    # TopK,
    # SAGPool,
    # # # DiffPool,
    EdgePool,
    GCN,
    GraphSAGE,
    GIN0,
    GIN,
    GlobalAttentionNet,
    Set2SetNet,
    SortPool,
    ASAP,
]


def logger(info):
    epoch =  info['epoch']
    val_loss, test_acc = info['val_loss'], info['val_acc']
    print('{:03d}: Val Loss: {:.4f}, Val Accuracy: {:.3f}'.format(
        epoch, val_loss, test_acc))


results = []

with open(args.train_split, 'r') as f:
    train_scenes = f.read().splitlines()
with open(args.val_split, 'r') as f:
    val_scenes = f.read().splitlines()

train_dataset = ScanNetDataset(args.root, train_scenes, split="train")
val_dataset = ScanNetDataset(args.root, val_scenes, split="val")

for Net in nets:
    best_result = (float('inf'), 0, 0)  # (loss, acc)
    best_model = (layers[0], hiddens[0])
    for num_layers, hidden in product(layers, hiddens):
        # dataset = get_dataset(dataset_name, sparse=Net != DiffPool)
        model = Net(train_dataset, num_layers, hidden)
        print('---------- \n - {}: {} num_layers, {} hidden'.format(Net.__name__, num_layers, hidden))
        loss, acc = train_eval(
            train_dataset,
            val_dataset,
            model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            lr_decay_factor=args.lr_decay_factor,
            lr_decay_step_size=args.lr_decay_step_size,
            weight_decay=0,
            # logger=logger,
            logger=None,
        )
        if acc < best_result[1]:
            best_result = (loss, acc)
            best_model = (num_layers, hidden)

    desc = '{:.3f}'.format(best_result[1])
    print('Best result - {}, {} num_layers, {} hidden'.format(desc, best_model[0], best_model[1]))
    results += ['- Model {}: {}'.format(Net.__name__, desc)]
print('-----\n{}'.format('\n'.join(results)))
