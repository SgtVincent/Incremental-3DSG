from logging import log
from networkx.algorithms.distance_measures import center
from torch.utils.tensorboard import SummaryWriter
import datetime 
import os 
from os.path import join, basename
import numpy as np
from scipy.spatial import KDTree
from timeit import default_timer as timer

# local import 
from SSG.utils.util_eva import EvalSceneGraph
from SSG.src.dataset_SGFN import dataset_loading_3RScan, load_mesh
from datasets.utils_3rscan import load_scene_graph, merge_centroids

import pathlib
PROJECT_LOGDIR = pathlib.Path(__file__).parent.parent.parent.absolute() # project dir 
DEFAULT_LOGDIR = join(PROJECT_LOGDIR, "tb_logs")
if not os.path.exists(DEFAULT_LOGDIR): os.makedirs(DEFAULT_LOGDIR)
DEFAULT_SSG_PATH = join(PROJECT_LOGDIR, "src", "SSG")

class RealtimeEvaluator3RScan():
    def __init__(self,
        scan_id,
        dataset_path="/media/junting/SSD_data/3RScan/3RScan", #"/media/junting/SSD_data/RIO_data",
        label_file="inseg.ply", # "labels.instances.annotated.ply" for RIO,  labels.instances.annotated.v2.ply for 3RScan
        class_file=join(DEFAULT_SSG_PATH, "data", "all_es", "classes160.txt"),
        relation_file=join(DEFAULT_SSG_PATH, "data", "all_es", "relationships.txt"), 
        annot_file=join(DEFAULT_SSG_PATH, "data", "all_es", "relationships_train.json"),
        logging=True,
        log_dir=DEFAULT_LOGDIR,
        flag_merge_centroids=False) -> None:

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

        # load classes 
        with open(class_file, "r") as f:
            self.object_names = f.read().splitlines()
        with open(relation_file, "r") as f:
            self.relation_names = f.read().splitlines()

        # load scene graph annotations 
        self.instances_id, self.obj_centroids_dict, self.obj_num_points, self.objects, self.relationships = \
            load_scene_graph(dataset_path, scan_id, label_file, annot_file)
        self.obj_centroids_arr = np.stack([self.obj_centroids_dict[id] for id in self.instances_id], axis=0) # numpy array 
        
        if flag_merge_centroids:
            self.instances_id, self.obj_centroids_arr, self.relationships, self.mapping_inst_id = \
                merge_centroids(self.instances_id, self.obj_centroids_arr, self.relationships, self.rel_same_part, num_points=self.obj_num_points)
            self.obj_centroids_dict = {self.instances_id[i]:self.obj_centroids_arr[i] for i in range(len(self.instances_id))}

        # process loaded annotations
        self.obj_kdtree = KDTree(self.obj_centroids_arr)
        self.num_instances = len(self.instances_id)
        self.rel_mat = np.full((self.num_instances, self.num_instances), len(self.relation_names), dtype=int)
        self.rel_count = np.zeros(len(self.relation_names))
        for rel in self.relationships:
            id_0, id_1, rel_label = rel[0], rel[1], rel[2]
            idx_0 = self.instances_id.index(id_0)
            idx_1 = self.instances_id.index(id_1)
            self.rel_mat[idx_0, idx_1] = rel_label
            self.rel_count[int(rel_label)] += 1

    def eval_objects(self, obj_centroids):
        centroids = list(obj_centroids.values())
        dist, idx = self.obj_kdtree.query(centroids)

        metric = {}
        metric["submaps_num"] = len(centroids)
        metric["submaps_acc"] = np.sum(np.array(dist) < self.obj_threshold) / float(len(centroids))

        return metric


    def eval_relations(self, edges, relations, obj_centroids):
        # # remove all "none" relations before eval
        # none_rel = len(self.relation_names)
        # valid_idx = [i for i in range(len(relations)) if relations[i] < none_rel]
        # keep none edges 
        valid_idx = list(range(len(relations)))
        metrics = {rel_name:{} for rel_name in self.relation_names}
        TP_counter = np.zeros(len(self.relation_names) +1) # none relation
        FP_counter = np.zeros(len(self.relation_names) +1) # none relation

        for i in valid_idx:
            edge = edges[i]
            rel = relations[i]
            obj_0 = obj_centroids[edge[0]]
            obj_1 = obj_centroids[edge[1]]
            dist, gt_idx = self.obj_kdtree.query([obj_0, obj_1])
            
            if np.any(np.array(dist) > self.obj_threshold): # objects position error 
                FP_counter[rel] += 1
            elif self.rel_mat[gt_idx[0], gt_idx[1]] != rel: # false positive
                FP_counter[rel] += 1
            else:
                TP_counter[rel] += 1
        
        # number of relations 
        
        # acc/precision + recall
        for i, rel_name in enumerate(self.relation_names):
            metrics[rel_name]["precision"] = float(TP_counter[i]) / float(TP_counter[i] + FP_counter[i] + 1e-6)
            metrics[rel_name]["recall"] = float(TP_counter[i]) / float(self.rel_count[i] + 1e-6)
        metrics["total"] = {}
        metrics["total"]["precision"] = float(np.sum(TP_counter)) / float(np.sum(TP_counter + FP_counter) + 1e-6)
        metrics["total"]["recall"] = float(np.sum(TP_counter)) / float(np.sum(self.rel_count) + 1e-6)

        return metrics

    def eval_obj(self, objects):
        pass

    # TODO: eval objects as well 
    def eval(self, obj_centroids, edges, relations):
        if len(obj_centroids) <= 0:
            return None
        # meta info 
        self.eval_count += 1
        if self.time_start is None:
            self.time_start = timer()
        time_now = timer()
        time_elapse = time_now - self.time_start
        x_axis = self.eval_count        
        if self.use_time_stamp:
            x_axis = time_elapse


        obj_metrics = self.eval_objects(obj_centroids)

        rel_metrics = self.eval_relations(edges, relations, obj_centroids)
        
        if self.logging:
            
            # write object metric 
            self.logger.add_scalars(
                "object/num_submaps",
                {'generated': obj_metrics["submaps_num"],
                'ground truth': len(self.instances_id)},
                x_axis)
            self.logger.add_scalar("object/acc", obj_metrics["submaps_acc"], x_axis)
            # write relationships metric
            for rel, v in rel_metrics.items():
                self.logger.add_scalar(f"precision/rel/{rel}", v["precision"], x_axis)
                self.logger.add_scalar(f"recall/rel/{rel}", v["recall"], x_axis)

        return rel_metrics

# TODO: implement this class for mass scans evaluation
class StaticEvaluator3RScan():
    def __init__(self,
        class_file, 
        relation_file) -> None:
                # load dataset meta info 
        self.class_file = class_file
        if not class_file: 
            self.class_file = join(DEFAULT_SSG_PATH, "data", "all_es", "classes160.txt")
        with open(self.class_file, "r") as f:
            self.class_names = f.read().splitlines()
        
        self.relation_file = relation_file
        if not relation_file:
            self.relation_file = join(DEFAULT_SSG_PATH, "data", "all_es", "relationships.txt")
        with open(self.relation_file, "r") as f:
            self.relation_names = f.read().splitlines()
        self.eval_tool = EvalSceneGraph(self.class_names, self.relation_names, multi_rel_prediction=False)

        pass