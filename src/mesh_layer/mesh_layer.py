import time
import numpy as np
import copy 
from threading import Lock
from visualization_msgs.msg import MarkerArray, Marker
from geometry_msgs.msg import Quaternion, Point, Pose
import rospy
import sys
sys.path.append("/home/junting/panoptic_ws/devel/lib/python3/dist-packages")
from voxblox_msgs.msg import MultiMesh

from utils.panoptic_utils import process_mesh
from utils.panoptic_utils import estimate_normal
from utils.panoptic_utils import rgba2rgb
class MeshSubmap:
    def __init__(self, frame_id, class_name, vertices, colors, normals, ros_time=False) -> None:

        # meta info
        # self.instance_number = kwargs.get("instance_number", -1)
        # self.instance_counter = kwargs.get("instance_counter", -1)
        self.frame_id = frame_id
        self.class_name = class_name
        self.updated = 1
        self.ros_time = ros_time
        if self.ros_time:
            self.live_time = rospy.get_time()
        else: 
            self.live_time = time.time()
        
        # mesh vertices 
        self.vertices = vertices
        self.colors = colors
        self.normals = normals

        
    def __len__(self):
        return len(self.vertices)

    def update(self, class_name, vertices, colors, normals):
        self.updated = 1
        if self.ros_time:
            self.live_time = rospy.get_time()
        else: 
            self.live_time = time.time()

        self.class_name = class_name
        self.vertices = vertices
        self.colors = colors
        self.normals = normals

    # def to_pcl(self, n=-1, method='uniform'):
    #     if method == 'uniform':
    #         all_pcls, idx = np.unique(self.vertices, axis=0, return_index=True)
    #         # set n < 0 to retrieve all points
    #         if n > 0 and n < all_pcls.shape[0]:
    #             idx = np.random.choice(idx, n, replace=len(idx)<n)
    #         pcls = self.vertices[idx,:]
    #         colors = self.colors[idx,:]
    #         normals = self.normals[idx,:]
            
    #     else: 
    #         print("Only uniform sampling methods implemented in class MeshSubmap!")
    #         raise NotImplementedError
        
    #     return pcls, colors, normals


class MeshLayer:
    
    def __init__(self):
        
        self.submap_dict = {}
        # meta info for scene graph prediction
        # mutex lock 
        self.mutex_lock = Lock()

    def __len__(self):
        return len(self.submap_dict)
    
    def process_multimesh(self, msg: MultiMesh):
    
        # convert MultiMesh to world frame coordinates
        vertices, colors, normals= process_mesh(msg.mesh)
        submap_id = int(msg.header.frame_id.split("_")[-1])
        class_name = msg.name_space[msg.name_space.find('_') + 1:] # '24_SM_Bed_lamp_b'
        if len(vertices) > 0:
            # create a submap if it does not yet exist.
            if submap_id not in self.submap_dict: 

                self.submap_dict[submap_id] = MeshSubmap(
                    submap_id,
                    class_name=class_name,
                    vertices=vertices,
                    colors=colors,
                    normals=normals
                    
                )
            else:
                self.submap_dict[submap_id].update(
                    class_name=class_name,
                    vertices=vertices,
                    colors=colors,
                    normals=normals,
                )

            return 1 # valid message
        else:
            if submap_id in self.submap_dict:
                # delete submap
                del self.submap_dict[submap_id]
                # delete edges
                num_edges = len(self.edge_indices)
                edge_idx_to_keep = [i for i in range(num_edges) if submap_id not in self.edge_indices[i]]
                self.edge_indices = [self.edge_indices[i] for i in edge_idx_to_keep]
                self.edge_relations = [self.edge_relations[i] for i in edge_idx_to_keep]

            return 0 # delete message

    def process_pointclouds(self, msg: Marker):

            vertices = np.array([[point.x, point.y, point.z] for point in msg.points], dtype=float)
            rgba_colors = np.array([[color.r, color.g, color.b, color.a] for color in msg.colors], dtype=float)
            colors = rgba2rgb(rgba_colors)
            

            submap_id = int(msg.header.frame_id.split("_")[-1])
            class_name = msg.ns # 'SM_Bed_lamp_b'
            
            if len(vertices) > 0 and class_name != "FreeSpace":
                # create a submap if it does not yet exist.
                if submap_id not in self.submap_dict: 
                    
                    normals = estimate_normal(vertices)
                    self.submap_dict[submap_id] = MeshSubmap(
                        submap_id,
                        class_name=class_name,
                        vertices=vertices,
                        colors=colors,
                        normals=normals
                    )
                    return 1
                # update submap if it has been updated 
                elif True:
                # elif len(vertices) != len(self.submap_dict[submap_id]):

                    normals = estimate_normal(vertices)
                    self.submap_dict[submap_id].update(
                        class_name=class_name,
                        vertices=vertices,
                        colors=colors,
                        normals=normals
                    )
                    return 1
                else:
                    self.submap_dict[submap_id].live_time = rospy.get_time()
            
            return 0
        




    




