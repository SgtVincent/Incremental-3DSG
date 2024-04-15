import json
from os.path import join, basename
import numpy as np
from torch.utils.data import dataset
import networkx as nx

from SSG.src.dataset_SGFN import dataset_loading_3RScan, load_mesh

def load_scene_graph(dataset_path, scan_id, label_file, annot_json):

    ################## load data ###############
    # load mesh data
    scan_path = join(dataset_path, scan_id)
    mesh_data = load_mesh(
        scan_path, 
        label_file=label_file,
        use_rgb=False,
        use_normal=False)
    vertices = mesh_data['points']
    instances = mesh_data['instances']
    instances_id = np.unique(instances)

    # load annotations
    with open(annot_json, "r") as f:
        data = json.load(f)

    ######### construct GT scene graph ###########
    try:
        scan_data = [scan for scan in data["scans"] if scan["scan"] == scan_id][0]
    except Exception as e:
        print(f"Scan ID not found in relationships file {annot_json}!")
        raise e

    objects = scan_data["objects"]
    relationships = scan_data["relationships"]
    instances_id = [id for id in instances_id if str(id) in objects]

    obj_centroids = {}
    obj_num_points = []
    for i, obj_id in enumerate(instances_id):
        if str(obj_id) in objects: 
            obj_centroids[obj_id] = np.mean(vertices[instances==obj_id], axis=0)
            obj_num_points.append(vertices[instances==obj_id].shape[0])
    
    return instances_id, obj_centroids, obj_num_points, objects, relationships


def merge_centroids(instances_id, centroids, relationships, merge_rel, num_points=None):
    
    if num_points is None: # all centroids have same weight 
        num_points = np.ones(len(instances_id))
    else:
        num_points = np.array(num_points)

    G = nx.Graph()
    G.add_nodes_from(instances_id)
    merge_relationships = [rel[:2] for rel in relationships if 
        rel[0] in instances_id and rel[1] in instances_id and rel[2] == merge_rel]
    G.add_edges_from(merge_relationships)

    # collect all components in graph connected by "same part" relation 
    components = [c for c in nx.connected_components(G)]

    # create old_id -> new_id mapping
    mapping_id = {old_id:min(c) for c in components for old_id in c}
    new_ids = sorted(list(set(mapping_id.values())))

    # calculate weighted merged centers
    new_centroids = {}
    for c in components:
        new_id = min(c)
        if len(c)==1:
            new_centroids[new_id] = centroids[instances_id.index(new_id)] # no change 
        else:
            c_indices = [instances_id.index(id) for id in c]
            old_centroids = centroids[c_indices,:]
            weights = num_points[c_indices]
            new_centroid = np.average(old_centroids, axis=0, weights=weights)
            new_centroids[new_id] = new_centroid

    new_centroids = np.stack([new_centroids[id] for id in new_ids], axis=0)
    
    # calculate new relationships
    new_relationships = []
    for rel in relationships:
        if rel[2] != merge_rel: # not "same part" relation
            # map old instance ids to new ids 
            new_rel = [mapping_id[rel[0]], mapping_id[rel[1]], rel[2], rel[3]]
            if new_rel not in new_relationships:
                new_relationships.append(new_rel)
    return new_ids, new_centroids, new_relationships, mapping_id