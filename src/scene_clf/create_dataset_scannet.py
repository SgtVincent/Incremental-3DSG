from clf_utils.scannet_data_utils import *
from clf_utils.dataset import RawScanNetDataset
import os

if __name__ == '__main__':

    ############# define path here ###############################
    root = "D:/datasets/ScanNet"
    scans_dir = "D:/datasets/ScanNet/scans"
    scene_type_mapping_file = "D:/datasets/ScanNet/scene_type_mapping.json"
    type_label_mapping_file = "D:/datasets/ScanNet/type_label_mapping.json"
    word_embedding_file = "data/scannet_glove.pickle"

    ############# create and save data ########################
    scenes = os.listdir(scans_dir)
    scannet_dataset = RawScanNetDataset(root,
                                        scenes,
                                        word_embedding_path=word_embedding_file,
                                        scene_type_mapping_file=scene_type_mapping_file,
                                        type_label_mapping_file=type_label_mapping_file)

    pass
