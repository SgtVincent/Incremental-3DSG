#!/usr/bin/env python
# python 3 

# common modules 
import time
import cv2
import numpy as np
import open3d as o3d
import scipy.ndimage as ndimage
from matplotlib import cm
# ros modules 
import rospy
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import PointCloud2, PointField
import sensor_msgs.point_cloud2 as pc2
# local modules 
# from utils.pcl_utils import ros_to_pcl_intensity
from utils.open3d_utils import convertCloudFromOpen3dToRos

FLAG_PUB = True
FLAG_SAVE = True


class RoomSegNode:

    def __init__(self, node_name='room_seg_node'):
        # ---------------------- ROS name/namespace ----------------- # 
        rospy.init_node(node_name)
        self.namespace = rospy.get_namespace().strip("/")
        self.node_name = node_name 

        # ----------------- get parameters from param server --------------- # 
        # self.param1 = rospy.get_param("PARAM_TOPIC")

        # --------------- custom initialization ------------------------- #
        self.frame = "world"

        # ----------------- initialize publisher & subscriber  ------------- # 
        self.sub = rospy.Subscriber('/panoptic_mapper/visualization/submaps/free_space_tsdf', PointCloud2, self.callback_free_space_tsdf, queue_size=1) # 
        self.pub = rospy.Publisher('/scene_graph/room_seg_node/room_segmentation', PointCloud2) 
    
        # ------------------ service handler ----------------------------- #
    
    
    def callback_free_space_tsdf(self, msg):

        field_names=[field.name for field in msg.fields]
        points_list = list(pc2.read_points(msg, skip_nans=True, field_names = field_names))
        points, intensity = np.array(points_list)[:,:3], np.array(points_list)[:,3]
        # filter out points with TSDF value < 0.01
        idx = intensity > 0.01
        free_space_points = points[idx, :]

        voxel_map = VoxelMap(free_space_points, 0.3)
        voxel_map.segment()

        pcd = voxel_map.convert_to_open3d_pcd()
        
        if FLAG_PUB:
            pub_msg = convertCloudFromOpen3dToRos(pcd, self.frame)
            self.pub.publish(pub_msg)

        if FLAG_SAVE:
            save_to_ply(pcd)
        
        return 

    def run(self):

        rospy.spin()
        return 

class VoxelMap:

    def __init__(self, points, voxel_size=0.3):
        
        self.min_coords = np.min(points, axis=0)
        self.voxel_size = voxel_size
        
        # calculate the grid map for voxels 
        coord_indices = np.round((points - self.min_coords)/voxel_size).astype(int) 
        self.coord_indices = coord_indices
        self.grid_size = np.max(coord_indices, axis=0) + 1
        self.binary_map = np.zeros(self.grid_size, dtype=int)
        self.binary_map[coord_indices[:,0], coord_indices[:,1], coord_indices[:,2]] = 1

    # TODO: add adaptive segmentation process to make sure 
    # output number of segments within the range of expected_num_segs
    def segment(self, erode_size=5, expected_num_segs=None):
        
        # 1. erode to get disjoined components  
        # eroded_map = ndimage.morphology.binary_erosion(self.binary_map, iterations=erode_size)
        eroded_map = ndimage.morphology.binary_erosion(
            self.binary_map, structure=np.ones((erode_size, erode_size, erode_size), dtype=np.int)
        )
        label_map, num_components = ndimage.measurements.label(eroded_map)
        labeled_binary_map = (label_map > 0)
        
        # 2. assign label to eroded points by label propagation 
        if num_components > 1: # need to propagate label 

            # while there are point clouds that has no label 
            while (np.any(np.logical_and(
                self.binary_map, 
                np.logical_xor(self.binary_map, label_map)
            ))):
                # use binary mask to ONLY update positions where label == 0 in last iter 
                # and assigned new label in this iter 
                fp = ndimage.generate_binary_structure(3,1)
                dilated_label_map = ndimage.morphology.grey_dilation(label_map, footprint=fp)
                propogated_mask = np.logical_xor(labeled_binary_map, (dilated_label_map > 0))
                label_map[propogated_mask] = dilated_label_map[propogated_mask]
                labeled_binary_map = (label_map > 0)
        
        else: # only 0 or 1 class exists after erosion 
            # assign label=1 to all point clouds
            label_map = np.copy(self.binary_map)

        self.label_map = label_map
        return self.label_map

    def convert_to_open3d_pcd(self, color_map="viridis"):
        
        if hasattr(self, 'label_map'): # save xyz, color data  
            coords = np.argwhere(self.binary_map == 1)
            points = coords * self.voxel_size + self.min_coords
            
            labels = self.label_map[coords[:,0], coords[:,1], coords[:,2]]
            # print(f"labels in room segmentation:{np.unique(labels)}")
            color_map = cm.get_cmap('viridis')
            colors = color_map(labels.astype(float) / np.max(labels)) # [N, 4], range [0,1]

            # create open3d PointCloud obj 
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            pcd.colors = o3d.utility.Vector3dVector(colors[:,:3])
            
        else: # only save xyz data 
            coords = np.argwhere(self.binary_map == 1)
            points = coords * self.voxel_size + self.min_coords

            # create open3d PointCloud obj 
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
        
        return pcd 


def save_to_ply(pcd, file_path="/home/junting/Downloads/dataset/large_flat/room_segmentation.ply"):
    
    o3d.io.write_point_cloud(file_path, pcd)
    return 


# def open3d_pcd_from_numpy(points, label):

#     pcd = o3d.geometry.PointCloud()
#     pcd.points = o3d.utility.Vector3dVector(points)
#     intensity = np.tile(intensity.reshape((-1,1)), (1,3))
#     pcd.colors = o3d.utility.Vector3dVector(intensity)

#     return pcd



if __name__=='__main__':

    node = RoomSegNode()
    node.run()
    
