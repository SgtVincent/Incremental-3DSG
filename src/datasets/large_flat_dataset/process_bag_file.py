import rosbag
import rospy 
import os 
from os.path import join, basename, exists
import numpy as np 
from visualization_msgs.msg import Marker
import pandas as pd 

# local import 
from mesh_layer.mesh_layer import MeshLayer

if __name__ == '__main__':

    rosbag_path = "/home/junting/Downloads/dataset/large_flat/2021-06-27-00-23-57_3.bag"
    gt_obj_df = pd.DataFrame(columns=['InstanceID', 'Name', 'Centroids'])
    data_dir = "./data"
    if not exists(data_dir):
        os.makedirs(data_dir)

    mesh_layer = MeshLayer()
    with rosbag.Bag(rosbag_path) as bag:
        # num_msgs = bag.get_message_count()
        # get the last message 
        msg_counter = 0
        for topic, msg, t in bag.read_messages():
            msg_counter += mesh_layer.process_pointclouds(msg)


