import networkx as nx
import torch
import numpy as np
import pandas as pd
import os
from torch_geometric.datasets import Planetoid
from torch_geometric.utils.convert import to_networkx

from clf_utils.dataset import ScanNetDataset

if __name__ == '__main__':
    # ------------------- data preparation -------------------- #
    root = "D:/datasets/ScanNet"
    scans_dir = "D:/datasets/ScanNet/scans"
    scenes = os.listdir(scans_dir)

    # pre_transform, transform = T.NormalizeScale(), T.SamplePoints(1024)
    pre_transform, transform = None, None
    dataset = ScanNetDataset(root, scenes, "all")

    pyg_graph = dataset[3]
    nx_graph = to_networkx(pyg_graph)

    import matplotlib.pyplot as plt
    plt.figure()

    nx.draw(nx_graph, cmap=plt.get_cmap('Set1'), node_size=75, linewidths=6)
    plt.show()
