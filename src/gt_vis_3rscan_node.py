#!/usr/bin/env python
# python 3 

# common modules 
import time
import cv2
import numpy as np
from numpy.core.defchararray import center
import open3d as o3d
import scipy.ndimage as ndimage
from matplotlib import cm
import pandas as pd
import pickle 
import json 
from os.path import join
# ros modules 
import rospy
from cv_bridge import CvBridge, CvBridgeError
from visualization_msgs.msg import MarkerArray, Marker
from geometry_msgs.msg import Quaternion, Pose, Point, Vector3
from std_msgs.msg import Header, ColorRGBA
from sensor_msgs.msg import PointCloud2, PointField
import sensor_msgs.point_cloud2 as pcl2
import struct

# local modules 
from SSG.src.dataset_SGFN import load_mesh
from datasets.utils_3rscan import merge_centroids

class GTVis3RScanNode:

    def __init__(self, node_name='gt_vis_3rscan_node'):
        rospy.init_node(node_name)
        self.node_name = node_name 
        self.class_file = "/home/junting/panoptic_ws/src/scene_graph/src/SSG/data/all_es/classes160.txt"
        self.relation_file = "/home/junting/panoptic_ws/src/scene_graph/src/SSG/data/all_es/relationships.txt"
        self.annot_json = "/home/junting/panoptic_ws/src/scene_graph/src/SSG/data/all_es/relationships_train.json"
        self.dataset_path = "/media/junting/SSD_data/3RScan/3RScan"
        self.scan_id = "0cac7578-8d6f-2d13-8c2d-bfa7a04f8af3"
        # label_file: ["labels.instances.align.annotated.v2.ply", "inseg.ply"]
        self.label_file= "inseg.ply"
        self.frame_id = "world"
        self.text_z_shift = 0.2

        self.topic_pcl2 = "/scene_graph/gt_vis_3rscan_node/pointclouds"
        self.topic_objects = "/scene_graph/gt_vis_3rscan_node/objects"
        self.topic_obj_names = "/scene_graph/gt_vis_3rscan_node/obj_names"
        self.topic_relations = "/scene_graph/gt_vis_3rscan_node/relations"
        self.topic_rel_names = "/scene_graph/gt_vis_3rscan_node/rel_names"

        # publisher of pointclouds2
        self.pub_pcl2 = rospy.Publisher(self.topic_pcl2, PointCloud2, queue_size=5)
        self.pub_objects = rospy.Publisher(self.topic_objects, MarkerArray, queue_size=1)
        self.pub_obj_names = rospy.Publisher(self.topic_obj_names, MarkerArray, queue_size=1)
        self.pub_relations = rospy.Publisher(self.topic_relations, MarkerArray, queue_size=1)
        self.pub_rel_names = rospy.Publisher(self.topic_rel_names, MarkerArray, queue_size=1)

        # publisher flags 
        self.flag_publish_pcls = False
        self.flag_publish_objects = True
        self.flag_publish_obj_names = True
        self.flag_publish_relations = True
        self.flag_publish_rel_names = True
        
        # advanced options
        self.integrate_submaps = True # whether to integrate objects connected by "same part"
        self.rel_same_part = 7 # "same part" relation label

    def prepare_data(self):

        ############ fetch original data in 3RScan ################## 
        # load classes 
        with open(self.class_file, "r") as f:
            self.object_names = f.read().splitlines()
        with open(self.relation_file, "r") as f:
            self.relation_names = f.read().splitlines()
        
        # load mesh 
        scan_path = join(self.dataset_path, self.scan_id)
        mesh_data = load_mesh(
            scan_path, 
            label_file=self.label_file,
            use_rgb=True,
            use_normal=False)

        self.points = mesh_data['points'][:, :3]
        self.colors = mesh_data['points'][:, 3:6]
        if self.label_file == "inseg.ply":
            self.colors = ((self.colors + 1.0) / 2.0 * 255.0).astype(int)
        self.instances = mesh_data['instances']
        instances_id = np.unique(self.instances)

        # load annotations
        with open(self.annot_json, "r") as f:
            data = json.load(f)
        try:
            scan_data = [scan for scan in data["scans"] if scan["scan"] == self.scan_id][0]
        except Exception as e:
            print(f"Scan ID not found in relationships file {self.annot_json}!")
            raise e

        self.objects = scan_data["objects"]
        self.relationships = scan_data["relationships"]
        
        ############# process data as required 
        # get object centroids 
        
        obj_centroids = []
        obj_num_points = []
        self.instances_id = []
        for i, obj_id in enumerate(instances_id):
            if str(obj_id) in self.objects: 
                obj_centroids.append(np.mean(self.points[self.instances==obj_id], axis=0))
                obj_num_points.append(self.points[self.instances==obj_id].shape[0])
                self.instances_id.append(obj_id)
                
        self.obj_centroids = np.stack(obj_centroids, axis=0)
        self.num_points = np.array(obj_num_points, dtype=int)

        if self.integrate_submaps:
            self.instances_id, self.obj_centroids, self.relationships, self.mapping_inst_id = \
                merge_centroids(self.instances_id, self.obj_centroids, self.relationships, self.rel_same_part, self.num_points)

        # create pointclouds2 message
        if self.flag_publish_pcls:
            self.pcl2_msg = self.create_pcl2_msg(self.points, self.colors)

        # create objects messages 
        if self.flag_publish_objects:
            self.objects_msg, self.obj_names_msg = self.create_objects_msg(self.flag_publish_obj_names)

        # create relation messages 
        if self.flag_publish_relations:
            self.relations_msg, self.relation_names_msg = self.create_relations_msg(self.flag_publish_rel_names)

        return 


    def create_pcl2_msg(self, pcls, colors, a=255):

        points = []
        for i in range(pcls.shape[0]):
            pcl = pcls[i]
            c = colors[i]
            rgb = struct.unpack('I', struct.pack('BBBB', c[2], c[1], c[0], a))[0]
            
            pt = [pcl[0], pcl[1], pcl[2], rgb]
            points.append(pt)

            fields = [
                PointField('x', 0, PointField.FLOAT32, 1),
                PointField('y', 4, PointField.FLOAT32, 1),
                PointField('z', 8, PointField.FLOAT32, 1),
                PointField('rgba', 12, PointField.UINT32, 1)
            ]

        header = Header()
        header.frame_id = self.frame_id
        pcl2_msg = pcl2.create_cloud(header, fields, points)
        
        return pcl2_msg

    def create_objects_msg(self, flag_publish_obj_names):

        marker_arr = MarkerArray()
        name_marker_arr = MarkerArray()
        class_names = [self.objects[str(id)] for id in self.instances_id]
        for i, id in enumerate(self.instances_id):
            
            centroid = self.obj_centroids[i]
            class_name = class_names[i]
            # centroid = (np.max(pcls, axis=0) + np.max(pcls, axis=0))/2.0 # select center of bbox 
            
            # if self.map_rawlabel2panoptic_id[obj.label] == 0:
            #     ns = "structure"
            #     scale=Vector3(0.5, 0.5, 0.5)
            #     color = ColorRGBA(1.0, 0.5, 0.0, 0.5)
            # else: # self.map_rawlabel2panoptic_id[obj.label] == 1 
            ns = class_name
            scale=Vector3(0.2, 0.2, 0.2)
            color = ColorRGBA(1.0, 0.0, 1.0, 0.5)
            
            marker = Marker(
                type=Marker.CUBE,
                id=id,
                ns=ns,
                # lifetime=rospy.Duration(2),
                pose=Pose(Point(*centroid), Quaternion(0,0,0,1)),
                scale=scale,
                header=Header(frame_id=self.frame_id),
                color=color,
                # text=submap.class_name,
                )

            marker_arr.markers.append(marker)

            if flag_publish_obj_names:
                text_pos = centroid + np.array([0.0,0.0,self.text_z_shift])
                name_marker = Marker(
                    type=Marker.TEXT_VIEW_FACING,
                    id=id,
                    ns=ns,
                    # lifetime=rospy.Duration(2),
                    pose=Pose(Point(*text_pos), Quaternion(0,0,0,1)),
                    scale=Vector3(0.2, 0.2, 0.2),
                    header=Header(frame_id=self.frame_id),
                    color=ColorRGBA(1.0, 0.0, 0.0, 0.8),
                    text=class_name)

                name_marker_arr.markers.append(name_marker)
        
        return marker_arr, name_marker_arr

    def create_relations_msg(self, flag_publish_rel_names):

        marker_arr = MarkerArray()

        name_marker_arr = MarkerArray()
        color_map = cm.get_cmap("viridis")
        centroids_dict = {self.instances_id[i]:self.obj_centroids[i] 
            for i in range(len(self.instances_id))}

        marker_arr = MarkerArray()
        
        for i, rel in enumerate(self.relationships):

            color = color_map(float(rel[2]) / len(self.relation_names))
            start_centroid = centroids_dict[rel[0]]
            end_centroid = centroids_dict[rel[1]]

            start_point = Point(*start_centroid) 
            end_point = Point(*end_centroid)   

            marker = Marker(
                type=Marker.ARROW,
                id=i,
                ns=rel[3],
                # lifetime=rospy.Duration(3),
                pose=Pose(Point(), Quaternion(0,0,0,1)),
                scale=Vector3(0.05, 0.1, 0.1),
                header=Header(frame_id=self.frame_id),
                color=ColorRGBA(*color),
                points=[start_point, end_point],
            )

            if flag_publish_rel_names:
                text_pos=(start_centroid + end_centroid)/2.0 + np.array((0,0,self.text_z_shift))
                name_marker = Marker( 
                    type=Marker.TEXT_VIEW_FACING,
                    id=i,
                    ns=rel[3],
                    # lifetime=rospy.Duration(3),
                    pose=Pose(Point(*text_pos), Quaternion(*(0,0,0,1))),
                    scale=Vector3(0.2, 0.2, 0.2),
                    header=Header(frame_id=self.frame_id),
                    color=ColorRGBA(1.0, 0.0, 0.0, 0.8),
                    text=rel[3])

                name_marker_arr.markers.append(name_marker)

            marker_arr.markers.append(marker)

        return marker_arr, name_marker_arr

    def run(self):
        self.prepare_data()

        rate = rospy.Rate(0.2)
        while not rospy.is_shutdown():
            # for msg in self.pcl2_msgs:
            #     self.pub_pcl2.publish(msg)
            if self.flag_publish_pcls:
                self.pub_pcl2.publish(self.pcl2_msg)
            if self.flag_publish_objects:
                self.pub_objects.publish(self.objects_msg)
            if self.flag_publish_obj_names:
                self.pub_obj_names.publish(self.obj_names_msg)
            if self.flag_publish_relations:
                self.pub_relations.publish(self.relations_msg)
            if self.flag_publish_rel_names:
                self.pub_rel_names.publish(self.relation_names_msg)
            rate.sleep()

if __name__=='__main__':

    scene_graph_vis_node = GTVis3RScanNode()
    scene_graph_vis_node.run()

