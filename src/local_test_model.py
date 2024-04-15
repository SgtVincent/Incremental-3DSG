#!/usr/bin/env python
# python 3

# common modules
import time
import cv2
from os.path import join, basename
import numpy as np
import open3d as o3d
import scipy.ndimage as ndimage
import sys
from matplotlib import cm
import pandas as pd
import pickle
import json
from threading import Lock
from scipy.spatial import KDTree
import torch 
import matplotlib.pyplot as plt

# local modules
from mesh_layer.sg_pred import SceneGraphPredictor

if __name__ == '__main__':

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # load model 
    sg_predictor = SceneGraphPredictor(rel_th=2.0)
    
    # load gt input data 
    data_file = "/home/junting/Downloads/dataset/model_input/data_SGFN.pkl"
    with open(data_file, "rb") as f:
        data = pickle.load(f)
    
    confidence = []

    for item in data: 
        points = torch.from_numpy(np.array(item['obj_points'])).to(device)
        edges = torch.from_numpy(np.array(item['edge_indices'])).to(device)
        descriptors = torch.from_numpy(np.array(np.array(item['descriptor']))).to(device)

        pred_obj_cls, pred_rel_cls, = sg_predictor.model(
            points, edges.contiguous(), descriptors, return_meta_data=False)
        
        confidence.append((torch.max(torch.exp(pred_rel_cls).detach().cpu() ,1)[0]).numpy())
    
    confidence = np.concatenate(confidence)
    plt.hist(confidence)
    plt.show()