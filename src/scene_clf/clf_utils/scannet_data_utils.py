import os
import numpy as np
import json
from plyfile import PlyData
import csv
from pathlib import Path

# get member values from scene0000_00.txt
def get_field_from_info_file(filename, field_name):
    lines = open(filename).read().splitlines()
    lines = [line.split(' = ') for line in lines]
    mapping = { x[0]:x[1] for x in lines }
    if field_name in mapping:
        return mapping[field_name]
    else:
        print('Failed to find %s in info file %s' % (field_name, filename))

# read positions of mesh vertices from .ply files
def read_mesh_vertices(filename):
    assert os.path.isfile(filename)
    with open(filename, 'rb') as f:
        plydata = PlyData.read(f)
        num_verts = plydata['vertex'].count
        vertices = np.zeros(shape=[num_verts, 3], dtype=np.float32)
        vertices[:,0] = plydata['vertex'].data['x']
        vertices[:,1] = plydata['vertex'].data['y']
        vertices[:,2] = plydata['vertex'].data['z']
    return vertices

# read data from .segs.json files
def read_segmentation(filename):
    assert os.path.isfile(filename)
    seg_to_verts = {}
    with open(filename) as f:
        data = json.load(f)
        num_verts = len(data['segIndices'])
        for i in range(num_verts):
            seg_id = data['segIndices'][i]
            if seg_id in seg_to_verts:
                seg_to_verts[seg_id].append(i)
            else:
                seg_to_verts[seg_id] = [i]
    return seg_to_verts, num_verts


# read data from .aggregation.json files
def read_aggregation(filename):
    assert os.path.isfile(filename)
    object_id_to_segs = {}
    label_to_segs = {}
    with open(filename) as f:
        data = json.load(f)
        num_objects = len(data['segGroups'])
        for i in range(num_objects):
            object_id = data['segGroups'][i]['objectId'] + 1 # instance ids should be 1-indexed
            label = data['segGroups'][i]['label']
            segs = data['segGroups'][i]['segments']
            object_id_to_segs[object_id] = segs
            if label in label_to_segs:
                label_to_segs[label].extend(segs)
            else:
                label_to_segs[label] = segs
    return object_id_to_segs, label_to_segs

def read_objects(filename):
    assert os.path.isfile(filename)
    with open(filename) as f:
        data = json.load(f)
        num_objects = len(data['segGroups'])
        objects = [data['segGroups'][i]['label']
            for i in range(num_objects)]
    return objects

def represents_int(s):
    try:
        int(s)
        return True
    except ValueError:
        return False


def read_label_mapping(filename, label_from='raw_category', label_to='nyu40id'):
    assert os.path.isfile(filename)
    mapping = dict()
    with open(filename) as csvfile:
        reader = csv.DictReader(csvfile, delimiter='\t')
        if label_to == "sequence":
            i = 0
            for row in reader:
                mapping[row[label_from]] = i
                i = i+1
        else:
            for row in reader:
                mapping[row[label_from]] = row[label_to]
    # if ints convert
    if represents_int(list(mapping.keys())[0]):
        mapping = {int(k):v for k,v in mapping.items()}
    return mapping

def read_scene_labels(scenes, scene_type_mapping_file, type_label_mapping_file):
    with open(scene_type_mapping_file, 'r') as f:
        scene_type_mapping = json.load(f)
    with open(type_label_mapping_file, 'r') as f:
        type_label_mapping = json.load(f)
    labels = [type_label_mapping[scene_type_mapping[scene]]
                for scene in scenes]
    num_class = len(type_label_mapping.keys())
    return np.array(labels, dtype=np.int), num_class


def get_objects(scans_dir, scenes):
    scans_dir = Path(scans_dir)
    objects = []
    for scene in scenes:
        scene_dir = scans_dir / scene
        agg_file = scene_dir / (scene + ".aggregation.json")
        objects = objects + read_objects(agg_file)
    return list(set(objects))

def get_object_vertices(scans_dir, scene):
    scans_dir = Path(scans_dir)

    # annotation file path
    mesh_file = scans_dir / scene / f"{scene}_vh_clean_2.ply"
    agg_file = scans_dir / scene / f"{scene}.aggregation.json"
    seg_file = scans_dir / scene / f"{scene}_vh_clean_2.0.010000.segs.json"
    # load annotations data from files
    mesh_vertices = read_mesh_vertices(mesh_file)
    object_id_to_segs, label_to_segs = read_aggregation(agg_file)
    seg_to_verts, num_verts = read_segmentation(seg_file)

    # collect point clouds for each object
    object_pcls = {}
    for object_id, segs in object_id_to_segs.items():
        vert_ids = []
        for seg in segs:
            vert_ids.extend(seg_to_verts[seg])
        vert_ids = np.unique(vert_ids)
        object_pcls[object_id-1] = mesh_vertices[vert_ids, :] # object_id starting from 1

    object_classes = read_objects(agg_file)

    return object_pcls, object_classes




