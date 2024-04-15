from torch_geometric.data import Data
import networkx as nx
import pickle

from torch_geometric.utils import to_networkx, from_networkx

from scipy.spatial import distance

from .scannet_data_utils import *
# TODO: add support for conversion between scannet raw data and networkx format data


def create_node_features(word_embedding_path,
                         objects,
                         positions=None,
                         other_features=None):

    assert(os.path.exists(word_embedding_path))
    with open(word_embedding_path, "rb") as f:
        word_embed = pickle.load(f)
    features = np.stack([word_embed[obj] for obj in objects], axis=0)
    if positions is not None:
        positions = np.reshape(positions, (features.shape[0], -1))
        features = np.concatenate((features, positions), axis=1)
    if other_features is not None:
        other_features = np.reshape(other_features, (features.shape[0], -1))
        features = np.concatenate((features, other_features), axis=1)

    return features


# TODO: improve the graph construction algorithm
# Ver 0.1:
# -- node features: word embedding
# -- edge weights: inverse of distance
# -- edge weights other choices: inverse of position difference vector
def create_graph_scannet(scans_dir,
                         scene,
                         word_embedding_path,
                         embed_pos=False,
                         embed_bbox=False,
                         max_deg=10,
                         directed=False,
                         min_dist=0.2,
                         ):

    object_id_to_vertices, object_classes = get_object_vertices(scans_dir, scene)
    positions = []
    for i in range(len(object_classes)):
        pos = np.mean(object_id_to_vertices[i], axis=0)
        positions.append(pos)

    positions = np.stack(positions, axis=0)

    # ---------------- extract node features ------------
    feature_pos = None
    feature_bbox = None
    if embed_pos:
        feature_pos = positions
    if embed_bbox:
        # TODO: implement bbox as node features
        raise NotImplementedError
    features = create_node_features(word_embedding_path, object_classes,
                                    positions=feature_pos, other_features=feature_bbox)

    # ---------------- extract edge weights ---------------
    dist_mat = distance.cdist(positions, positions)
    # set diag to DTYPE_MAX to cancel self loop
    dist_mat = dist_mat + np.eye(len(object_classes)) * np.finfo(dist_mat.dtype).max
    # clip minimum distance to clip the maximum edge weight
    dist_mat[dist_mat < min_dist] = min_dist
    # the closer the nodes are, the larger the edge weight should have
    weights_mat = 1 / dist_mat
    # calculate the top N=max_deg in edges that from the closest neighbors
    adj_mat = np.zeros(dist_mat.shape, dtype=int)
    edge_idx = np.argsort(dist_mat, axis=1)
    for i in range(len(object_classes)):
        adj_mat[i, edge_idx[i,:max_deg]] = 1
    # WARNING: if graph is undirected, max_deg < real deg < 2*max_deg (out edges and in edges all kept)
    if directed is False:
        adj_mat = adj_mat + np.transpose(adj_mat)

    # convert adj_mat and weights_mat to format required by pyg
    # need to represent undirected graph by inserting two edges: a-->b, b-->a
    edges = [] # N*2
    edge_weights = [] # N
    for i in range(len(object_classes)):
        for j in range(len(object_classes)):
            if adj_mat[i, j] > 0:
                edges.append([i, j])
                edge_weights.append(weights_mat[i,j])
    edges = np.transpose(np.array(edges, dtype=int))
    edge_weights = np.array(edge_weights).reshape((edges.shape[1], -1))

    return features, edges, edge_weights, positions
