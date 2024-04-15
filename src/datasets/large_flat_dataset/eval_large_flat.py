from logging import log
from networkx.algorithms.distance_measures import center
from torch.utils.tensorboard import SummaryWriter
import datetime 
import os 
from os.path import join, basename
import numpy as np
import open3d as o3d 
from scipy.spatial import KDTree
from timeit import default_timer as timer

# local import 
from SSG.utils.util_eva import EvalSceneGraph
from SSG.src.dataset_SGFN import dataset_loading_3RScan, load_mesh
from datasets.utils_3rscan import load_scene_graph, merge_centroids

import pathlib
PROJECT_LOGDIR = pathlib.Path(__file__).parent.parent.parent.parent.absolute() # project dir 
DEFAULT_LOGDIR = join(PROJECT_LOGDIR, "tb_logs")
if not os.path.exists(DEFAULT_LOGDIR): os.makedirs(DEFAULT_LOGDIR)
DEFAULT_SSG_PATH = join(PROJECT_LOGDIR, "src", "SSG")

map_classname2label = {"Bookstore_Library": 0, "Laundry_Room": 1, "Closet": 2, "Classroom": 3, 
    "Storage_Basement_Garage": 4, "Apartment": 5, "Stairs": 6, "Conference_Room": 7, 
    "Dining_Room": 8, "Bathroom": 9, "Living_room_Lounge": 10, "Gym": 11, "Copy_Mail_Room": 12, 
    "ComputerCluster": 13, "Hallway": 14, "Bedroom_Hotel": 15, "Lobby": 16, "Kitchen": 17, 
    "Office": 18, "Misc.": 19, "Game_room": 20}

class RealtimeEvaluatorLargeFlat():
    def __init__(self,
        dataset_path="/home/junting/Downloads/dataset/large_flat",
        label_file="large_flat.ply", 
        #TODO: add evaluation file after annotation finished 
        class_file=join(DEFAULT_SSG_PATH, "data", "all_es", "classes160.txt"),
        relation_file=join(DEFAULT_SSG_PATH, "data", "all_es", "relationships.txt"), 
        annot_file=join(DEFAULT_SSG_PATH, "data", "all_es", "relationships_train.json"),
        logging=True,
        log_dir=DEFAULT_LOGDIR) -> None:

        # parameters
        self.obj_threshold = 0.5
        self.eval_count = 0
        self.time_start = None # initialized when first calling eval()
        self.rel_same_part = 7 # "same part" relation label
        self.use_time_stamp = True # use time stamp as x-axis of evaluation plots
        
        # set logger
        self.logging = logging
        if self.logging:
            self.log_dir = join(log_dir, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
            self.logger = SummaryWriter(self.log_dir)
        
        # load submap GT data 
        self.submap_dir = join(dataset_path, "objects_segmentation")
        self.num_submaps = len(os.listdir(self.submap_dir))
        
        # load segmentation GT data 
        self.segmentation_dir = join(dataset_path, "room_segmentation")
        self.gt_room_segs = []
        for seg_name in os.listdir(self.segmentation_dir):
            class_name = seg_name[:seg_name.rfind('_')]
            label = map_classname2label[class_name]
            seg_file = join(self.segmentation_dir, seg_name)
            pcd_load = o3d.io.read_point_cloud(seg_file)
            points = np.asarray(pcd_load.points)
            min_corner = np.min(points, axis=0)
            max_corner = np.max(points, axis=0)
            # assume z does not affect IOU result 
            bbox = [min_corner[0], min_corner[1], max_corner[0], max_corner[1]]
            self.gt_room_segs.append({
                "label": label,
                "bbox": bbox
            })

    def eval_roomlayer(self, room_points, room_labels):
        
        all_iou_scores = []
        correct_pred_count = 0
        for i in range(len(room_labels)):
            min_corner = np.min(room_points[i], axis=0)
            max_corner = np.max(room_points[i], axis=0)
            # assume z does not affect IOU result 
            bbox = [min_corner[0], min_corner[1], max_corner[0], max_corner[1]]
            
            iou_scores = [
                bb_intersection_over_union(bbox, self.gt_room_segs[j]["bbox"])
                for j in range(len(self.gt_room_segs))
            ]
            max_iou_idx = np.argmax(iou_scores)
            all_iou_scores.append(iou_scores[max_iou_idx])
            if self.gt_room_segs[max_iou_idx]["label"] == room_labels[i]:
                correct_pred_count += 1
        
        avg_iou_score = np.mean(all_iou_scores)
        acc = correct_pred_count / float(len(room_labels))
        return avg_iou_score, acc
            

    def eval(self, room_points, room_labels):
        if len(room_labels) <= 0:
            return 
        # meta info 
        self.eval_count += 1
        if self.time_start is None:
            self.time_start = timer()
        time_now = timer()
        time_elapse = time_now - self.time_start
        x_axis = self.eval_count        
        if self.use_time_stamp:
            x_axis = time_elapse

        avg_iou, acc = self.eval_roomlayer(room_points, room_labels)

        if self.logging:
            print("writing to tensorboard...")
            # write object metric 
            self.logger.add_scalars(
                "RoomLayer/num_rooms",
                {'generated': len(room_labels),
                'ground truth': len(self.gt_room_segs)},
                x_axis)
            self.logger.add_scalar("RoomLayer/average IOU", avg_iou, x_axis)
            self.logger.add_scalar("RoomLayer/acc", acc, x_axis)

        return 


def bb_intersection_over_union(boxA, boxB):
	# determine the (x, y)-coordinates of the intersection rectangle
	xA = max(boxA[0], boxB[0])
	yA = max(boxA[1], boxB[1])
	xB = min(boxA[2], boxB[2])
	yB = min(boxA[3], boxB[3])
	# compute the area of intersection rectangle
	interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
	# compute the area of both the prediction and ground-truth
	# rectangles
	boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
	boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
	# compute the intersection over union by taking the intersection
	# area and dividing it by the sum of prediction + ground-truth
	# areas - the interesection area
	iou = interArea / float(boxAArea + boxBArea - interArea)
	# return the intersection over union value
	return iou  