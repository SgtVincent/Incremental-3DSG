# asldoc-2021-st-juntingchen
## Incremental 3D Scene Graph Construction
This system constructs a hierarchical 3D scene graph from dense mapping for high-level planning incrementally in real time. ![figure](images/demo_scene_graph_construction.gif).
## Environment Setup
### Prerequisite
Since this system runs on top of the dense mapper. You should first configure the environment for [panoptic_mapping](https://github.com/ethz-asl/panoptic_mapping/tree/release/beta).

### Minor changes in panoptic_mapping 
Please first move the RGB pointclouds publisher [source file](src/panoptic_mapper_pclpub_node.cpp) to `./panoptic_mapping_ros/app` in [panoptic_mapping](https://github.com/ethz-asl/panoptic_mapping/tree/release/beta) package and modify [cmake file](https://github.com/ethz-asl/panoptic_mapping/blob/release/beta/panoptic_mapping_ros/CMakeLists.txt) correspondingly to build this node. 
### Python Setup
Now we assume you have already installed [ROS](http://wiki.ros.org/ROS/Installation), created ROS working space, and successfully run the panoptic mapper.

**Common Libs**

`pip3 install numpy scipy pandas json opencv-python`

**Point Cloud Libs**

Follow the official instructions and install [pcl](https://github.com/strawlab/python-pcl), [open3d](http://www.open3d.org/docs/release/getting_started.html). 

**Deep Learning Libs**

Follow the official instructions and instal [Pytorch](https://pytorch.org/get-started/locally/), [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html). Pytorch versions `1.8.0/1.9.0`, `cuda111/cpu` have been tested. 



### Environment information
The environment setup has only been tested on the following system

> Ubuntu 20.04  
> gcc 9.3.0  
> python 3.8.10  
> CUDA 11.1  

If you meet some unknown errors, feel free to raise an issue.

## Data Preparation 

### Model 
First, you should download pre-trained relationship prediction model from [3DSSG](https://github.com/ShunChengWu/3DSSG), or directly download from the [link](https://drive.google.com/file/d/1a2q7yMNNmEpUfC1_5Wuor0qDM-sBStFZ/view). Then you should unzip file, and put all `*.pth` files to `./src/SSG/CVPR21` directory.

Then, you should download the room scans from [here](). It contains all RGBD input, modified labels, and room segmentation annotations, etc. There are two scans in this zip file, `flat` and `large_flat`. 
## Run the system 
Please Change all data paths in the [launch file](./launch/run.launch) to your local path. Then with panoptic mapper running at the background, you could simply run `roslaunch scene_graph run.launch` to start the system. 