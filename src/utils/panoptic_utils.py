import numpy as np
from std_msgs.msg import Header, ColorRGBA
from visualization_msgs.msg import MarkerArray, Marker
from sensor_msgs.msg import PointCloud2, PointField
from voxblox_msgs.msg import Mesh, MeshBlock
import open3d as o3d


def calculate_mesh(mesh_msg: Mesh):
    
    vertices = []
    block_edge_length = mesh_msg.block_edge_length

    for mesh_block in mesh_msg.mesh_blocks:
        # Each vertex is given as its distance from the blocks origin in units of
        # (2*block_size), see mesh_vis.h for the slightly convoluted
        # justification of the 2.
        point_conv_factor = 2.0 / np.iinfo(np.uint16).max
        index = mesh_block.index

        mesh_x = (np.array(mesh_block.x, dtype=float) * point_conv_factor +
            float(index[0])) * block_edge_length
        mesh_y = (np.array(mesh_block.y, dtype=float) * point_conv_factor +
            float(index[1])) * block_edge_length
        mesh_z = (np.array(mesh_block.z, dtype=float) * point_conv_factor +
            float(index[2])) * block_edge_length

        block_vertices = np.stack((mesh_x, mesh_y, mesh_z), axis=-1)
        vertices.append(block_vertices)

    vertices = np.concatenate(vertices, axis=0)

    return vertices


def estimate_normal(vertices):
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(vertices)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    return np.asarray(pcd.normals)

def get_color(mesh_msg: Mesh):
    
    colors = []
    for mesh_block in mesh_msg.mesh_blocks:
        r = np.array(list(bytearray(mesh_block.r)), dtype=int)
        g = np.array(list(bytearray(mesh_block.g)), dtype=int)
        b = np.array(list(bytearray(mesh_block.b)), dtype=int)
        block_colors = np.stack((r,g,b), axis=-1)
        colors.append(block_colors)

    colors = np.concatenate(colors, axis=0)

    return colors 

def rgba2rgb(rgba, background=(255.0,255.0,255.0)):
    
    N, ch = rgba.shape
    assert ch == 4
    rgb = np.zeros((N, 3), dtype=float)
    r, g, b, a = rgba[:,0], rgba[:,1], rgba[:,2], rgba[:,3]
    R, G, B = background

    rgb[:,0] = r * a * 255.0 + (1.0 - a) * R
    rgb[:,1] = g * a * 255.0 + (1.0 - a) * G
    rgb[:,2] = b * a * 255.0 + (1.0 - a) * B

    return np.asarray(rgb, dtype=int)



def process_mesh(mesh_msg: Mesh):

    vertices = calculate_mesh(mesh_msg)
    if vertices.shape[0] <= 0:
        return [], [], [] 
    # TODO: check how to generate normals correctly
    normals = estimate_normal(vertices)
    colors = get_color(mesh_msg)

    return vertices, colors, normals 
