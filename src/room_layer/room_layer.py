import numpy as np
import copy 
from scipy.spatial import cKDTree

# ros import 
from visualization_msgs.msg import MarkerArray, Marker
from geometry_msgs.msg import Quaternion, Point, Pose
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2

# from utils.open3d_utils import 
# local import 

class RoomNode:
    
    def __init__(self, grid_points, room_idx, grid_size, **kwargs):
        # required attributes 
        self.grid_points = grid_points 
        self.room_idx = room_idx
        self.grid_size = grid_size
        # optinal attributes
        self.object_ids = kwargs.get('object_ids',None)
        self.room_label = kwargs.get('room_label', 19) # default: 19:"Misc."

        self.pos = np.mean(self.grid_points, axis=0)

        return 

    def update_objects(self, object_ids):

        if self.object_ids is None:
            self.object_ids = np.unique(object_ids)
        else:
            self.object_ids = np.unique(np.concatenate(self.object_ids, object_ids))
        return 

    def update_room_label(self, label):
        self.room_label = label

class RoomLayer:
    
    def __init__(self, grid_points, grid_labels, grid_size):
        
        self.grid_points = np.array(grid_points)
        self.grid_labels = np.array(grid_labels)
        self.grid_size = grid_size
        self.grid_kd_tree = cKDTree(self.grid_points)

        self.room_ids = np.unique(self.grid_labels)
        self.room_nodes = []

        for idx in self.room_ids:
            self.room_nodes.append(RoomNode(
                grid_points=self.grid_points[self.grid_labels == idx, :],
                room_idx=idx,
                grid_size=grid_size
            ))

        return 

    def __len__(self):
        return len(self.room_nodes)

    def get_room_by_idx(self,idx):
        idx = np.argwhere(self.room_ids == idx)[0][0]
        return self.room_nodes[idx]

    def query_points_rooms(self, points):

        dists, indices = self.grid_kd_tree.query(points)
        return self.grid_labels[indices]

    def add_objects(self, positions, obj_ids):

        positions = np.array(positions)
        obj_ids = np.array(obj_ids)
        
        obj_room_ids = self.query_points_rooms(positions)

        for idx in range(len(self.room_nodes)):
            label = self.room_ids[idx]
            room_node = self.room_nodes[idx]
            obj_ids_of_room = np.array(obj_ids)[obj_room_ids == label]
            room_node.update_objects(obj_ids_of_room)
        
        return 

    @classmethod
    def from_msg(self, msg: PointCloud2, grid_size=0.3):
        # Get cloud data from ros_cloud
        field_names=[field.name for field in msg.fields]
        cloud_data = list(pc2.read_points(msg, skip_nans=True, field_names = field_names))

        # Check empty
        if len(cloud_data)==0:
            print("Converting an empty cloud")
            return None

        # convert to numpy 
        room_grid_points = np.array([pt[:3] for pt in cloud_data])
        room_grid_rgb = np.array([pt[3] for pt in cloud_data])
        _, room_grid_labels = np.unique(room_grid_rgb, return_inverse=True) 
        room_grid_labels += 1 # room labels starting from 1
        return self(room_grid_points, room_grid_labels, grid_size)

