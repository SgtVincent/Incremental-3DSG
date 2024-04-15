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

# ros modules
import rospy
from cv_bridge import CvBridge, CvBridgeError
from visualization_msgs.msg import MarkerArray, Marker
from geometry_msgs.msg import Quaternion, Pose, Point, Vector3
from std_msgs.msg import Header, ColorRGBA
from sensor_msgs.msg import PointCloud2, PointField
import sys
sys.path.append("/home/junting/panoptic_ws/devel/lib/python3/dist-packages")
from voxblox_msgs.msg import MultiMesh
# import sensor_msgs.point_cloud2 as pc2
# local modules
from mesh_layer.mesh_layer import MeshLayer
from object_layer.object_layer import ObjectLayer
from mesh_layer.sg_pred import SceneGraphPredictor
# SSG
from SSG.utils.util_label import getLabelMapping
from datasets.eval_3rscan import RealtimeEvaluator3RScan



class SceneGraphPredictNode:

    def __init__(self, node_name='scene_graph_predict_node'):
        # ---------------------- ROS name/namespace ----------------- #
        rospy.init_node(node_name)
        self.namespace = rospy.get_namespace().strip("/")
        self.node_name = node_name

        # ----------------- get parameters from param server --------------- #
        # self.param1 = rospy.get_param("PARAM_TOPIC")

        # ----------------- TODO: set parameters from launch file ---------------#
        self.object_elavate_node = 3.0
        self.object_elavate_text = 0.5
        self.room_elavate_node = 8.0
        self.room_elavate_text = 1.0

        # --------------- custom initialization ------------------------- #
        self.frame = "world"
        self.rel_dist_thresh = 2.0
        self.rel_clf_thresh = 0.3
        self.msg_counter = 0
        self.vis_objects = True
        self.vis_relations = True
        self.vis_object_name = True
        self.vis_rel_name = True
        self.vis_mode = "inplace"
        self.scan_id = "0cac7578-8d6f-2d13-8c2d-bfa7a04f8af3"
        self.realtime_eval = False
        self.merge_centroids = True
        self.mesh_outdated_time = 10000


        self.topic_multimesh = "/panoptic_mapper/visualization/mesh"
        self.topic_pcls = "/panoptic_mapper/submap_points"
        self.topic_objects = "/scene_graph/scene_graph_predict_node/object_nodes"
        self.topic_obj_names = "/scene_graph/scene_graph_predict_node/object_names"
        self.topic_relations = "/scene_graph/scene_graph_predict_node/relations"
        self.topic_rel_names = "/scene_graph/scene_graph_predict_node/rel_names"
        label_names, label_name_mapping, label_id_mapping = getLabelMapping("ScanNet20")
        with open("/home/junting/panoptic_ws/src/scene_graph/src/SSG/data/all_es/relationships.txt", "r")  as f:
            self.relation_names = f.read().splitlines()
        # TODO: move edges to object layer 
        self.edge_indices = []
        self.edge_relations = []

        self.sg_predictor = SceneGraphPredictor(self.rel_dist_thresh)
        self.mesh_layer = MeshLayer()
        if self.realtime_eval:
            self.evaluator = RealtimeEvaluator3RScan(
                self.scan_id, 
                flag_merge_centroids=self.merge_centroids)

        # ----------------- initialize publisher & subscriber  ------------- #
        # TODO: compose topic name from namespace and node name
        # self.sub_multimesh = rospy.Subscriber(
        #     self.topic_multimesh, MultiMesh, self.callback_multimesh, queue_size=50) #
        self.sub_pcls = rospy.Subscriber(
            self.topic_pcls, Marker, callback=self.callback_pointclouds, queue_size=1)

        # TODO: compose topic name from namespace and node name
        self.pub_relations = rospy.Publisher(self.topic_relations, MarkerArray, queue_size=1)
        self.pub_rel_names = rospy.Publisher(self.topic_rel_names, MarkerArray, queue_size=1)
        self.pub_object_nodes = rospy.Publisher(self.topic_objects, MarkerArray, queue_size=1) 
        self.pub_object_names = rospy.Publisher(self.topic_obj_names, MarkerArray, queue_size=1) 
        # ------------------ service handler ----------------------------- #

    def callback_pointclouds(self, msg: Marker, update_step=5):
        # update submaps in mesh_layer and set update flag for each submap
        
        with self.mesh_layer.mutex_lock:
            # self.mesh_layer.process_msg(msg)
            # self.msg_counter += 1
            self.msg_counter += self.mesh_layer.process_pointclouds(msg)
            
            if self.msg_counter % update_step == 0:

                self.update_scene_graph()

                # evaluation
                if self.realtime_eval:
                    self.evaluator.eval(
                        self.centroids_dict,
                        self.edge_indices, 
                        self.edge_relations,
                        )
        return 1

    # TODO: fuse SGFN predicted object label with semantic segmantation label
    def publish_objects(self, vis_mode="inplace", flag_vis_name=True):
        assert vis_mode in ["elavated", "inplace"]
        with self.mesh_layer.mutex_lock:
            marker_arr = MarkerArray()
            if flag_vis_name:
                name_marker_arr = MarkerArray()
            for id, submap in self.mesh_layer.submap_dict.items():
                
                submap_id = f"submap_{id}"
                pcls = submap.vertices
                centroid = np.mean(pcls, axis=0) # select mean 
                # centroid = (np.max(pcls, axis=0) + np.max(pcls, axis=0))/2.0 # select center of bbox 
                if vis_mode == "elavated":
                    centroid = centroid + np.array([0.0,0.0,self.object_elavate_node])
                
                # if self.map_rawlabel2panoptic_id[obj.label] == 0:
                #     ns = "structure"
                #     scale=Vector3(0.5, 0.5, 0.5)
                #     color = ColorRGBA(1.0, 0.5, 0.0, 0.5)
                # else: # self.map_rawlabel2panoptic_id[obj.label] == 1 
                ns = "object"
                scale=Vector3(0.2, 0.2, 0.2)
                color = ColorRGBA(0.0, 1.0, 0.5, 0.5)
                
                marker = Marker(
                    type=Marker.CUBE,
                    id=id,
                    ns=ns,
                    # lifetime=rospy.Duration(2),
                    pose=Pose(Point(*centroid), Quaternion(0,0,0,1)),
                    scale=scale,
                    header=Header(frame_id=submap_id),
                    color=color,
                    text=submap.class_name,
                    )

                marker_arr.markers.append(marker)

                if flag_vis_name:
                    text_pos = centroid + np.array([0.0,0.0,self.object_elavate_text])
                    name_marker = Marker(
                        type=Marker.TEXT_VIEW_FACING,
                        id=id,
                        ns=ns,
                        # lifetime=rospy.Duration(2),
                        pose=Pose(Point(*text_pos), Quaternion(0,0,0,1)),
                        scale=Vector3(0.2, 0.2, 0.2),
                        header=Header(frame_id=submap_id),
                        color=ColorRGBA(0.0, 0.0, 0.0, 0.8),
                        text=submap.class_name)

                    name_marker_arr.markers.append(name_marker)
            
            self.pub_object_nodes.publish(marker_arr)
            if flag_vis_name:
                self.pub_object_names.publish(name_marker_arr)
            
            return 



    def publish_relations(self, vis_mode="inplace", color_map='viridis', vis_name=True):
        assert vis_mode in ["elavated", "inplace"]

        # avoid data racing on edges
        with self.mesh_layer.mutex_lock:
            edges = self.edge_indices
            relations = self.edge_relations

            marker_arr = MarkerArray()
            if vis_name:
                name_marker_arr = MarkerArray()
            color_map = cm.get_cmap(color_map)

            # colors = color_map(labels.astype(float) / max(self.room_layer.room_ids))
            submaps = [submap for id, submap in self.mesh_layer.submap_dict.items()]
            centroids_dict = {}
            for submap in submaps:
                pcls = submap.vertices
                centroids_dict[submap.frame_id] = np.mean(pcls, axis=0) # select mean 
                # centroids_dict[submap.frame_id] = (np.max(pcls, axis=0) + np.max(pcls, axis=0))/2.0 # select center of bbox 

            marker_arr = MarkerArray()
            
            for i, edge in enumerate(edges):
                # point_elavated = np.concatenate((room.pos[:2], [elavate_z]))
                rel = relations[i]
                # relation=8: none 
                if rel >= len(self.relation_names):
                    continue
                color = color_map(float(rel) / len(self.relation_names))
                if vis_mode == "elavated":
                    start_centroid = centroids_dict[edge[0]]+np.array((0.0,0.0,self.object_elavate_node))
                    end_centroid = centroids_dict[edge[1]]+np.array((0.0,0.0,self.object_elavate_node))
                else: # "inplace":
                    start_centroid = centroids_dict[edge[0]]
                    end_centroid = centroids_dict[edge[1]]

                start_point = Point(*start_centroid) 
                end_point = Point(*end_centroid)   

                marker = Marker(
                    type=Marker.ARROW,
                    id=i,
                    ns=self.relation_names[rel],
                    lifetime=rospy.Duration(3),
                    # pose=Pose(Point(), Quaternion(0,0,0,1)),
                    scale=Vector3(0.05, 0.1, 0.1),
                    header=Header(frame_id=self.frame),
                    color=ColorRGBA(*color),
                    points=[start_point, end_point],
                )

                if vis_name:
                    text_pos=(start_centroid + end_centroid)/2.0 + \
                        np.array((0,0,self.object_elavate_text-self.object_elavate_node))
                    name_marker = Marker( 
                        type=Marker.TEXT_VIEW_FACING,
                        id=i,
                        lifetime=rospy.Duration(3),
                        pose=Pose(Point(*text_pos), Quaternion(*(0,0,0,1))),
                        scale=Vector3(0.3, 0.3, 0.3),
                        header=Header(frame_id=self.frame),
                        color=ColorRGBA(0.0, 0.0, 0.0, 0.8),
                        text=self.relation_names[rel])

                    name_marker_arr.markers.append(name_marker)

                marker_arr.markers.append(marker)

        self.pub_relations.publish(marker_arr)
        if vis_name:
            self.pub_rel_names.publish(name_marker_arr)
        return

    def update_scene_graph(self):

        # check timer to delete timeout submaps
        submap_ids = list(self.mesh_layer.submap_dict.keys())
        for submap_id in submap_ids:
            # delete submap if outdated 
            submap = self.mesh_layer.submap_dict[submap_id]
            if rospy.get_time() - submap.live_time > self.mesh_outdated_time:
    
                del self.mesh_layer.submap_dict[submap_id]
                # delete edges
                num_edges = len(self.edge_indices)
                edge_idx_to_keep = [i for i in range(num_edges) if submap_id not in self.edge_indices[i]]
                self.edge_indices = [self.edge_indices[i] for i in edge_idx_to_keep]
                self.edge_relations = [self.edge_relations[i] for i in edge_idx_to_keep]
                print(f"delete submap {submap_id}")


        updated_submaps = [submap for id, submap in self.mesh_layer.submap_dict.items()
            if submap.updated == 1 ]
        updated_ids = [submap.frame_id for submap in updated_submaps]

        # calculate centroids 
        self.centroids_dict = {}
        for id, submap in self.mesh_layer.submap_dict.items():
            pcls = submap.vertices
            self.centroids_dict[id] = np.mean(pcls, axis=0)
        centroids_arr = [c for id, c in self.centroids_dict.items()]
        if len(centroids_arr) > 0:
            self.centroids_kdtree = KDTree(centroids_arr)

        if len(updated_ids) <= 0:
            # print("Not received any mesh message ...")
            pass
        else:
            # collect all submaps neiboring to updated submaps to update relationships 
            updated_centroids = [self.centroids_dict[id] for id in updated_ids]
            list_of_subneighbors_idx = self.centroids_kdtree.query_ball_point(updated_centroids, r=self.rel_dist_thresh)
            neighbors_idx = set([idx for sub_list in list_of_subneighbors_idx for idx in sub_list])
            
            all_ids = [id for id, c in self.centroids_dict.items()]
            neighbors_ids = [all_ids[idx] for idx in neighbors_idx]
            neighbors_submaps = [self.mesh_layer.submap_dict[id] for id in neighbors_ids]

            # with self.edge_mutex_lock:
            # edges, pred_rel_label, pred_rel_prob = \
            #     self.sg_predictor.predict(updated_submaps)
            edges, pred_rel_label, pred_rel_confidence = \
                self.sg_predictor.predict(neighbors_submaps)
            # TODO: try to use slideing window average on relation prediction
            # TODO: try to reject relation prediction with the probability
            for i, edge in enumerate(edges):
                if pred_rel_confidence[i] > self.rel_clf_thresh:
                    submap_edge = [neighbors_ids[edge[0]], neighbors_ids[edge[1]]]
                    if submap_edge in self.edge_indices:
                        j = self.edge_indices.index(submap_edge)
                        self.edge_relations[j] = pred_rel_label[i]
                    else:
                        self.edge_indices.append(submap_edge)
                        self.edge_relations.append(pred_rel_label[i])
        return 

    # TODO: Solve the racing write/read mesh_layer problem 
    # in unsynchronized sub/pub mode 
    def run(self):

        rate = rospy.Rate(2)

        while not rospy.is_shutdown():            

            ######################  visualization #################
            # try:
            if len(self.mesh_layer) > 0:
                if self.vis_objects:
                    self.publish_objects(vis_mode=self.vis_mode)
                if self.vis_relations:
                    self.publish_relations(vis_mode=self.vis_mode, vis_name=self.vis_rel_name)
                
            # except:
                # print("write/read racing in publish_relations!")

            rate.sleep()

        # rospy.spin()
        return


if __name__ == '__main__':
    scene_graph_vis_node = SceneGraphPredictNode()
    # rospy.spin()
    scene_graph_vis_node.run()