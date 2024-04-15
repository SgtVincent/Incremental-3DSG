from clf_utils.word_embedding_utils import create_embedding
from clf_utils.scannet_data_utils import *
import os

if __name__ == '__main__':
    scans_dir = "D:/datasets/ScanNet/scans"
    out_file = "data/scannet_glove.pickle"
    scenes = os.listdir(scans_dir)
    objects = get_objects(scans_dir, scenes)
    create_embedding(objects, out_file)