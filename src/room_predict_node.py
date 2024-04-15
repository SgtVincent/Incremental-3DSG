#!/usr/bin/env python
# python 3 

# common modules 
import time
import cv2
import numpy as np
import pcl 
import open3d as o3d
import scipy.ndimage as ndimage
from matplotlib import cm
import pandas as pd
import pickle 
import json 
# ros modules 
import rospy
from cv_bridge import CvBridge, CvBridgeError
from visualization_msgs.msg import MarkerArray, Marker
from geometry_msgs.msg import Quaternion, Pose, Point, Vector3
from std_msgs.msg import Header, ColorRGBA
from sensor_msgs.msg import PointCloud2, PointField
# import sensor_msgs.point_cloud2 as pc2

# local modules 
from object_layer.object_layer import ObjectLayer
from room_layer.room_layer import RoomLayer
from datasets.large_flat_dataset.eval_large_flat import RealtimeEvaluatorLargeFlat

class RoomPredictionNode:

    def __init__(self, node_name='room_predict_node'):
        # ---------------------- ROS name/namespace ----------------- # 
        rospy.init_node(node_name)
        self.namespace = rospy.get_namespace().strip("/")
        self.node_name = node_name 

        # ----------------- get parameters from param server --------------- # 
        # self.param1 = rospy.get_param("PARAM_TOPIC")

        # ----------------- TODO: set parameters from launch file ---------------# 
        self.object_elavate_node = 0.0
        self.object_elavate_text = 1.0
        self.room_elavate_node = 2.0
        self.room_elavate_text = 3.0
        self.vis_mode = "inplace"
        # --------------- custom initialization ------------------------- #
        self.frame = "world"
        self.object_layer = None
        self.room_layer = None
        self.realtime_eval = True

        #  self.topic_bounding_volumes = "/panoptic_mapper/visualization/bounding_volumes"
        self.topic_objects = "/scene_graph/scene_graph_predict_node/object_nodes"
        self.topic_relations = "/scene_graph/scene_graph_predict_node/relations"
        self.topic_room_segmentation = "/scene_graph/room_seg_node/room_segmentation"
        
        self.topic_room_nodes = '/scene_graph/room_predict_node/room_nodes'
        self.topic_room_names = '/scene_graph/room_predict_node/room_names'
        self.topic_room_seg_to_pub = '/scene_graph/room_predict_node/room_segmentation'
        self.topic_edges_room2object = '/scene_graph/room_predict_node/edges_room2object'
    
        # files path 
        self.label_file = rospy.get_param("~label_file", "/home/junting/Downloads/dataset/flat/labels_with_nyu40.csv")

        self.room_clf = pickle.load(
            open("/home/junting/panoptic_ws/src/scene_graph/src/scene_clf/data/clf_histogram.pickle", "rb"))

        # load ground truth label mapping 
        # label_csv = pd.read_csv("/home/junting/Downloads/dataset/flat/labels_with_nyu40.csv", sep=',')
        # label_csv = pd.read_csv("/home/junting/Downloads/dataset/large_flat/labels_with_nyu40.csv", sep=',')
        label_csv = pd.read_csv(self.label_file, sep=',')
        label_csv['ClassID'] = label_csv['ClassID'].astype(int)
        label_csv['nyu40id'] = label_csv['nyu40id'].astype(int)

        self.map_rawlabel2nyu40id = pd.Series(label_csv.nyu40id.values,index=label_csv.ClassID).to_dict()
        self.map_rawlabel2panoptic_id = pd.Series(label_csv.PanopticID.values,index=label_csv.ClassID).to_dict()
        self.map_classname2rawlabel = pd.Series(label_csv.ClassID.values,index=label_csv.Name).to_dict()
        self.hist_len = 41

        with open("/media/junting/Elements/ScanNet/type_label_mapping.json", "r") as f:
            map_class2roomlabel = json.load(f)
            self.map_roomlabel2class = dict((v,k) for k,v in map_class2roomlabel.items())

        if self.realtime_eval:
            self.evaluator = RealtimeEvaluatorLargeFlat()

        # ----------------- initialize publisher & subscriber  ------------- # 
        # TODO: compose topic name from namespace and node name
        # self.sub_bounding_volumes = rospy.Subscriber(
        #     '/panoptic_mapper/visualization/bounding_volumes', MarkerArray, self.callback_bounding_volumes, queue_size=1) # 
        # self.pub_object_nodes = rospy.Publisher('/scene_graph/scene_graph_vis_node/object_nodes', MarkerArray) 
        # self.pub_object_names = rospy.Publisher('/scene_graph/scene_graph_vis_node/object_names', MarkerArray) 
        self.pub_room_nodes = rospy.Publisher(self.topic_room_nodes, MarkerArray, queue_size=10) 
        self.pub_room_names = rospy.Publisher(self.topic_room_names, MarkerArray, queue_size=10) 
        self.pub_room_segmentation = rospy.Publisher(self.topic_room_seg_to_pub, MarkerArray, queue_size=10)
        self.pub_edges_room2object = rospy.Publisher(self.topic_edges_room2object, MarkerArray, queue_size=10)
        # TODO: finish room layer publish code 
        # self.pub_room_layer = rospy.Publisher('/scene_graph/scene_graph_vis_node/room_layer', MarkerArray) 
    
        # ------------------ service handler ----------------------------- #
    
    # def callback_bounding_volumes(self, msg: MarkerArray):
    #     # update objectlayer and marker array together in cb function 
    #     # In this way, publish() function only reads and avoids caused racing writes 

    #     self.object_layer = ObjectLayer.from_msg(msg)
    #     self.publish_object_layer()
    #     return 
    
    def prepare_data(self, timeout=1):
        # TODO: change this method to call rosservice 
        try: 
            obj_msg = rospy.wait_for_message(
                self.topic_objects,MarkerArray, timeout=timeout)
        except rospy.exceptions.ROSException as e:
            print(f"Not received {self.topic_objects} within {timeout} secs, continue...")
            obj_msg = None
        # TODO: change this method to call rosservice 
        try:
            rseg_msg = rospy.wait_for_message(
                self.topic_room_segmentation,PointCloud2, timeout=timeout)
        except rospy.exceptions.ROSException as e:
            print(f"Not received {self.topic_room_segmentation} within {timeout} secs, continue...")
            rseg_msg = None

        return obj_msg, rseg_msg

        # except rospy.exceptions.ROSException as e:
        #     print(f"Not received {topic} within {timeout} secs, continue...")
        #     return None


    def publish_object_layer(self, vis_mode="inplace", flag_vis_name=True):
        assert vis_mode in ["elavated", "inplace"]

        marker_arr = MarkerArray()
        if flag_vis_name:
            name_marker_arr = MarkerArray()
        for obj_idx in self.object_layer.obj_ids:
            
            obj = self.object_layer.obj_dict[obj_idx]
            point = np.array(obj.pos)

            if vis_mode == "elavated":
                point = point + np.array([0.0,0.0,self.object_elavate_node])
            
            if self.map_rawlabel2panoptic_id[obj.label] == 0:
                ns = "structure"
                scale=Vector3(0.5, 0.5, 0.5)
                color = ColorRGBA(1.0, 0.5, 0.0, 0.5)
            else: # self.map_rawlabel2panoptic_id[obj.label] == 1 
                ns = "object"
                scale=Vector3(0.2, 0.2, 0.2)
                color = ColorRGBA(0.5, 1.0, 0.0, 0.5)
            
            marker = Marker(
                type=Marker.SPHERE,
                id=obj.id,
                ns=ns,
                # lifetime=rospy.Duration(2),
                pose=Pose(Point(*point), Quaternion(*obj.orient)),
                scale=scale,
                header=Header(frame_id=self.frame),
                color=color,
                # text=text,
                )

            marker_arr.markers.append(marker)

            if flag_vis_name:
                text_pos = point + np.array([0.0,0.0,self.object_elavate_text])
                name_marker = Marker(
                    type=Marker.TEXT_VIEW_FACING,
                    id=obj.id,
                    ns=ns,
                    # lifetime=rospy.Duration(2),
                    pose=Pose(Point(*text_pos), Quaternion(*obj.orient)),
                    scale=Vector3(0.2, 0.2, 0.2),
                    header=Header(frame_id=self.frame),
                    color=ColorRGBA(0.0, 0.0, 0.0, 0.8),
                    text=obj.name)

                name_marker_arr.markers.append(name_marker)
        
        self.pub_object_nodes.publish(marker_arr)
        if flag_vis_name:
            self.pub_object_names.publish(name_marker_arr)
        
        return 

    def publish_room_layer(self, vis_mode="inplace", flag_vis_name=True, flag_vis_room_seg=True, color_map='viridis'):
        assert vis_mode in ["elavated", "inplace"]
        color_map = cm.get_cmap(color_map)
        marker_arr = MarkerArray()

        if flag_vis_name:
            name_marker_arr = MarkerArray()

        if flag_vis_room_seg:
            room_seg_marker_arr = MarkerArray()

        for room in self.room_layer.room_nodes:
            
            point_elavated = np.concatenate((room.pos[:2], [self.room_elavate_node]))
            color = color_map(float(room.room_idx) / max(self.room_layer.room_ids))
            marker = Marker(
                type=Marker.CUBE,
                ns=self.map_roomlabel2class[room.room_label],
                id=room.room_idx,
                lifetime=rospy.Duration(3),
                pose=Pose(Point(*point_elavated), Quaternion(0,0,0,1)),
                scale=Vector3(0.2, 0.2, 0.2),
                header=Header(frame_id=self.frame),
                color=ColorRGBA(*color),
                # text=text,
                )

            marker_arr.markers.append(marker)

            if flag_vis_name:
                point_elavated = np.concatenate((room.pos[:2], [self.room_elavate_text]))
                room_class_name = self.map_roomlabel2class[room.room_label]
                name_marker = Marker(
                    type=Marker.TEXT_VIEW_FACING,
                    ns=self.map_roomlabel2class[room.room_label],
                    id=room.room_idx,
                    lifetime=rospy.Duration(3),
                    pose=Pose(Point(*point_elavated), Quaternion(0,0,0,1)),
                    scale=Vector3(0.3, 0.3, 0.3),
                    header=Header(frame_id=self.frame),
                    color=ColorRGBA(0.0, 0.0, 0.0, 0.8),
                    text=f"{room_class_name}_{room.room_idx}")

                name_marker_arr.markers.append(name_marker)
        
            if flag_vis_room_seg:
                grid_points = [Point(*point) for point in room.grid_points]
                room_seg_color = list(color)
                room_seg_color[3] *= 0.8
                free_space_marker = Marker(
                    type=Marker.SPHERE_LIST,
                    ns=self.map_roomlabel2class[room.room_label],
                    id=room.room_idx,
                    lifetime=rospy.Duration(3),
                    points=grid_points,
                    pose=Pose(Point(0,0,0), Quaternion(0,0,0,1)),
                    scale=Vector3(0.05, 0.05, 0.05),
                    header=Header(frame_id=self.frame),
                    color=ColorRGBA(*room_seg_color))

                room_seg_marker_arr.markers.append(free_space_marker)

        self.pub_room_nodes.publish(marker_arr)
        if flag_vis_name:
            self.pub_room_names.publish(name_marker_arr)
        if flag_vis_room_seg:
            self.pub_room_segmentation.publish(room_seg_marker_arr)
        return 
    
    def publish_edges_room2object(self, vis_mode="inplace", color_map='viridis'):
        assert vis_mode in ["elavated", "inplace"]

        marker_arr = MarkerArray()
        color_map = cm.get_cmap(color_map)
        # colors = color_map(labels.astype(float) / max(self.room_layer.room_ids))
        for room in self.room_layer.room_nodes:
            
            # point_elavated = np.concatenate((room.pos[:2], [elavate_z]))
            color = color_map(float(room.room_idx) / max(self.room_layer.room_ids))
            room_point = Point(room.pos[0], room.pos[1], self.room_elavate_node)

            marker = Marker(
                type=Marker.LINE_LIST,
                id=room.room_idx,
                lifetime=rospy.Duration(3),
                # pose=Pose(Point(*point_elavated), Quaternion(0,0,0,1)),
                scale=Vector3(0.05, 0.0, 0.0),
                header=Header(frame_id=self.frame),
                color=ColorRGBA(*color),
                # text=text,
            )

            room_objects = self.object_layer.get_objects_by_ids(room.object_ids)
            # LINE_LIST marker displays line between points 0-1, 1-2, 2-3, 3-4, 4-5
            for obj in room_objects:
                marker.points.append(room_point)
                obj_pos = obj.pos + np.array([0.0,0.0,self.object_elavate_node])
                marker.points.append(Point(*obj_pos))

            marker_arr.markers.append(marker)
        
        self.pub_edges_room2object.publish(marker_arr)
        return 

        

    def run(self):

        rate = rospy.Rate(1)
        
        while not rospy.is_shutdown():

            ###############  process messages  ###################
            obj_msg, rseg_msg = self.prepare_data()
            if obj_msg:
                self.object_layer = ObjectLayer.from_msg(obj_msg, self.map_classname2rawlabel)
            if rseg_msg:
                self.room_layer = RoomLayer.from_msg(rseg_msg)
                if self.object_layer is not None:
                    positions = self.object_layer.get_positions(self.object_layer.obj_ids)
                    self.room_layer.add_objects(positions, self.object_layer.obj_ids)

            ################### integrate data  #####################
            # predict room class 
            if self.object_layer is not None and self.room_layer is not None:
                for room in self.room_layer.room_nodes:
                    obj_ids = room.object_ids
                    if len(obj_ids > 0):
                        obj_labels = self.object_layer.get_labels(obj_ids)
                        nyu40_labels = [self.map_rawlabel2nyu40id[label] for label in obj_labels]
                        hist = np.bincount(np.array(nyu40_labels), minlength=self.hist_len)
                        room_label = self.room_clf.predict(hist.reshape((1,-1)))[0]
                        room.update_room_label(room_label)

            if self.realtime_eval:
                room_points = [node.grid_points for node in self.room_layer.room_nodes]
                room_labels = [node.room_label for node in self.room_layer.room_nodes]
                self.evaluator.eval(room_points, room_labels)

            ######################  visualization #################     
            # scene_graph_predict_node publishes object layer
            # if self.object_layer is not None:
            #     self.publish_object_layer(vis_mode=self.vis_mode)
            if self.room_layer is not None:
                # print(f"rooms in scene: {self.room_layer.room_ids}")
                self.publish_room_layer(vis_mode=self.vis_mode)
            if self.object_layer is not None and self.room_layer is not None:
                self.publish_edges_room2object(vis_mode=self.vis_mode)

            rate.sleep()
        
        
        # rospy.spin()
        return 

if __name__=='__main__':

    scene_graph_vis_node = RoomPredictionNode()
    scene_graph_vis_node.run()