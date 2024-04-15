# classify by objects occurences
import sklearn
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
import pickle

from clf_utils.scannet_data_utils import *


def create_hist(scans_dir, scenes, label_map_file):
    scans_dir = Path(scans_dir)
    # label_map = read_label_mapping(label_map_file, label_from='raw_category', label_to='sequence')
    label_map = read_label_mapping(label_map_file, label_from='raw_category', label_to='nyu40id')
    labels = [int(label) for label in label_map.values()]
    max_label = max(labels)
    hist_vectors = []
    for scene in scenes:
        scene_dir = scans_dir / scene
        agg_file = scene_dir / (scene + ".aggregation.json")
        objects = read_objects(agg_file)
        label_ids = [int(label_map[obj]) for obj in objects]
        hist_vec = np.bincount(np.array(label_ids), minlength=max_label+1)
        hist_vectors.append(hist_vec)

    return np.stack(hist_vectors, axis=0)


if __name__ == "__main__":
    ############# define path here ###############################
    # scans_dir = "D:/datasets/ScanNet/scans"
    # scene_type_mapping_file = "D:/datasets/ScanNet/scene_type_mapping.json"
    # type_label_mapping_file = "D:/datasets/ScanNet/type_label_mapping.json"
    # label_map_file = "D:/datasets/ScanNet/scannetv2-labels.combined.tsv"
    scans_dir = "/media/junting/Elements/ScanNet/scans"
    scene_type_mapping_file = "/media/junting/Elements/ScanNet/scene_type_mapping.json"
    type_label_mapping_file = "/media/junting/Elements/ScanNet/type_label_mapping.json"
    label_map_file = "/media/junting/Elements/ScanNet/scannetv2-labels.combined.tsv"
    ############# load data here ###################################
    scenes = os.listdir(scans_dir)
    with open("/home/junting/panoptic_ws/src/scene_graph/src/scene_clf/split/train.txt", 'r') as f:
        train_scenes = f.read().splitlines()
    with open("/home/junting/panoptic_ws/src/scene_graph/src/scene_clf/split/val.txt", 'r') as f:
        val_scenes = f.read().splitlines()

    train_hist = create_hist(scans_dir, train_scenes, label_map_file)
    val_hist = create_hist(scans_dir, val_scenes, label_map_file)
    train_labels, num_class = read_scene_labels(train_scenes, scene_type_mapping_file, type_label_mapping_file)
    val_labels, _ = read_scene_labels(val_scenes, scene_type_mapping_file, type_label_mapping_file)

    ############## train model ########################
    clf = RandomForestClassifier()
    clf.fit(train_hist, train_labels)

    #############  val model ########################
    val_score = clf.score(val_hist, val_labels)
    print(f"validation score:{val_score}")

    ##### train and save model with all training data/validation data #####

    save_path = "/home/junting/panoptic_ws/src/scene_graph/src/scene_clf/data/clf_histogram.pickle"
    all_scenes = train_scenes + val_scenes
    all_hist = create_hist(scans_dir, all_scenes, label_map_file)
    all_labels, num_class = read_scene_labels(all_scenes, scene_type_mapping_file, type_label_mapping_file)
    clf = RandomForestClassifier()
    clf.fit(train_hist, train_labels)
    pickle.dump(clf, open(save_path, 'wb'))
