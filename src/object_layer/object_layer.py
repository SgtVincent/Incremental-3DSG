from visualization_msgs.msg import MarkerArray, Marker
from geometry_msgs.msg import Quaternion, Point, Pose
import numpy as np
import copy 

class ObjectNode:
    
    def __init__(self, name, id, label, pos, orient, **kwargs):
        # required attributes 
        self.name = name
        self.id = id
        self.label = label
        self.pos = pos # [x,y,z]
        self.orient = orient # [x,y,z,w]
        # self.size = size # [x,y,z]
        # optinal attributes 
        self.class_name = kwargs.get('class_name',"unknown")


class ObjectLayer:
    
    def __init__(self, obj_ids, obj_list):
        
        # self.obj_ids = []
        # self.obj_dict = {}
        self.obj_ids = obj_ids
        self.obj_dict = {}
        for i in range(len(obj_ids)):
            self.obj_dict[obj_ids[i]] = obj_list[i]
        

    def __len__(self):
        return len(self.obj_ids)

    def get_objects_by_ids(self,ids):
        return [self.obj_dict[id] for id in ids]

    def get_positions(self, ids):
        pos = [self.obj_dict[id].pos for id in ids]
        return pos
    
    def get_labels(self, ids):
        labels = [self.obj_dict[id].label for id in ids]
        return labels
    
    @classmethod
    def from_msg(self, msg: MarkerArray, name_label_mapping):
        
        obj_ids = []
        obj_list = []
        for marker in msg.markers:

            submap_id = marker.id
            class_name = marker.text
            class_id = name_label_mapping[class_name]
            
            pos = marker.pose.position
            pos_l = [pos.x, pos.y, pos.z]
            orient = marker.pose.orientation
            orient_l = [orient.x, orient.y, orient.z, orient.w]

            obj = ObjectNode(class_name, submap_id, class_id, 
                pos_l, orient_l)

            obj_ids.append(submap_id)
            obj_list.append(obj)

        return self(obj_ids, obj_list)




