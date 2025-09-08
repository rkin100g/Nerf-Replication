# Nerf-Replication

Codebase for replicating the paper NeRF from ZJU CAD & CG Lab

## Setup

```sh
# Create conda environment
conda create -n nerf-rep python=3.10
conda activate nerf-rep

# Install PyTorch according to your environment
# Be sure you have CUDA installed, we use CUDA 12.1 in our experiments
# NOTE: you need to make sure the CUDA version used to compile the torch is the same as the version you installed
# Use `nvcc -V` to check the CUDA version
pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121

# Install other dependencies
pip install -r requirements.txt
```

## Data preparation

Download NeRF synthetic dataset and add a link to the data directory. After preparation, you should have the following directory structure:
We have provied the minimal `lego` dataset for you in the [Github release page](https://github.com/pengsida/project_page_assets/releases/download/nerf-replication/lego.zip).

```
data/nerf_synthetic
|-- chair
|   |-- test
|   |-- train
|   |-- val
|-- drums
|   |-- test
......
```

## NeRF Replication

### Configurations

The configuration file are stored in `configs/nerf`, which contains the parameters for replicating NeRF.
You can adjust the parameters according to your preference.

Parameter description:
```yaml
task: "nerf_replication"
gpus: [1] # set gpu device number
exp_name: "nerf"
scene: "lego"
trained_model_dir: "data/trained_model"

# set path for each modules
train_dataset_module: src.datasets.nerf.blender
test_dataset_module: src.datasets.nerf.blender
network_module: src.models.nerf.network
renderer_module: src.models.nerf.renderer.volume_renderer
loss_module: src.train.trainers.nerf
evaluator_module: src.evaluators.nerf

...
```

The default parameters can be found in `src/config/config.py`.

### Dataset

Dataset file path: `src/datasets/nerf/blender.py`

You need to implement the dataset class, which is responsible for loading the data and providing it to the model.
The core functions include: `__init__`, `__getitem__`, and `__len__`.
- `__init__`: This function is responsible for loading the specified format file from disk, calculating and storing it in a specific form.
- `__getitem__`: This function is responsible for providing the input required for training and the ground truth output to the model at runtime. For example, for NeRF, it provides 1024 rays and 1024 RGB values.
- `__len__`: This function returns the number of training or testing samples. The index value obtained from `__getitem__` is usually in the range [0, len-1].

After implementing the dataset class, you can use the following command to debug it:

```sh
python run.py --type dataset --cfg_file configs/nerf/lego.yaml
```

### Model

> [!CAUTION]
> This module has been implemented, don't modify the model file.

Model file path: `src/models/nerf/network.py`

The model module is responsible for defining the model structure and the forward propagation process.
The core functions include: `__init__`, `forward`.
- `__init__`: This function is responsible for defining the model structure, including the number of layers, the number of neurons in each layer, and the activation function.
- `forward`: This function is responsible for the forward propagation process, which includes the input of the dataset and the output of the model.
For NeRF, the input is a batch of rays, and the output is a batch of RGB values.

After implementing the network class, you can use the following command to debug it:
```sh
python run.py --type network --cfg_file configs/nerf/lego.yaml
```

### Renderer

Renderer file path: `src/models/nerf/renderer/volume_renderer.py`

The renderer module is responsible for rendering the output of the model.
The core functions include: `__init__`, `render`.
- `__init__`: This function is responsible for defining the rendering parameters, including the number of samples, the step size, and the background color.
- `render`: This function is responsible for rendering the output of the model, which includes the RGB values and the depth values.
The detailed rendering process is described in the paper.

### Trainer and Evaluator

Trainer file path: `src/train/trainers/nerf.py`
Evaluator file path: `src/evaluators/nerf.py`

The trainer module is responsible for training the model, including the loss function and the optimizer.
The evaluator module is responsible for evaluating the model, including the PSNR and SSIM metrics.
This two modules are simple and easy to implement, so here we only provide the file path.

```sh
python train.py --cfg_file configs/nerf/lego.yaml
```

```sh
python run.py --type evaluate --cfg_file configs/nerf/lego.yaml
```

### Inference

> [!CAUTION]
> Don't modify the model file.

Considering that you are limited by the GPU memory, we provide a pretrained model for you to test the inference process.
You can download the pretrained model from the [Github release page](https://github.com/pengsida/project_page_assets/releases/download/nerf-replication/latest.pth) and put in in the `data/trained_model` directory.

The pretrained model path should be like this: `data/trained_model/task_name/scene_name/exp_name/latest.pth`.

Then, run the following command to test the inference process:
```sh
python run.py --type evaluate --cfg_file configs/nerf/lego.yaml
```

# Experiment Record
Following are the record of progress and problems encontered in the experiments

## Structure and Core of Nerf
Representation of Scene using Nerf :
We represent a static scene as a continuous 5D function that outputs the radiance emitted in each direction (θ, φ) at each point (x, y, z) in space, and a density at each point which acts like a difffferential opacity controlling how much radiance is accumulated by a ray passing through (x, y, z).

Process of Nerf:
1) march camera rays through the scene to generate a sampled set of 3D points
2) use those points and their corresponding 2D viewing directions as input to the neural network to produce an output set of colors and densities
    3D coordinate x  -> 8 fully-connected layers(using ReLU activations and 256channels per layer)
    -> σ and a 256-dimensional feature vector -> concatenated with the camera ray’s viewing direction
    -> 1 fully-connected layer(using a ReLU activation and 128 channels) -> RGB color
3) use classical volume rendering techniques to accumulate those colors and densities into a 2D image
   

input: a set of images with known camera poses
input of network: 5D coordinate(saptial location(x,y,z) and viewing direction(θ, φ))a 3D Cartesian unit vector d as direction
output:a single volume density and view-dependent RGB color.

## Set up
I have updated environment from cuda 11.5 to cuda 12.1
done

## Data preparation
Dataset file path: `src/datasets/nerf/blender.py`

I have downloaded single sceen "lego",directory structure is as follows:
```
data/nerf_synthetic/lego
|-- test
|   |-- r_0_depth_0001.png
|   |-- r_0_normal_0001.png
|   |-- r_0_png
|   |-- ...
|   |-- r_199.png
|-- train
|   |-- r_0.png
|   |-- ...
|   |-- r_99.png
|-- val
|   |-- r_0.png
|   |-- ...
|   |-- r_99.png
|-- transforms_train.json
|-- transforms_val.json
|-- transforms_test.json
```

format of transforms_tain/val/test.json:
```
"camera_angle_x"
"frames": [
    {
            "file_path": "./train/r_0",
            "rotation":
            "transform_matrix": 
    },
    ...
]
```

## Implement of Dataset class
Dataset file path: `src/datasets/nerf/blender.py`
### Input and Output
Input：
```
cfg.test/train_dataset including data_root,split,input_ratio,cams,H,W
```
Output：
```
data_dict ={
    "colors":torch.from_numpy(colors).float(),
    "rays_o":torch.from_numpy(rays_o).float(),
    "rays_d":torch.from_numpy(rays_d).float(),
    "depth":torch.from_numpy(depth).float(),     # only if split is test
    "normal":torch.from_numpy(normal).float(),   # only if split is test
}
```
### Data flow and Structure
**overall data flow:**
```
cfg(data_root) -> _init_() -> self.data(data of each frame including image, transform_matrix), self.camera_angle_x -> _getitem_() -> data_dict={"colors", "ray_o", "ray_d"}
```

**_getitem_() backward reasoning:** 
```
colors -> pixels(u,v) -> uniformaly random choose 
ray_o -> camera_xyz in world coordinate -> T
ray_d -> direction vector in world coordinate -> trasnfrom from camera coordinate to world coordinate -> direction vector in camera coordiante -> K -> f,cx,cy -> camera_angle_x 
```

### Explanation of key formulas 
**Computation of K:**

[已知视场角下求解相机内参矩阵](https://blog.csdn.net/weixin_49053303/article/details/140603386?ops_request_misc=&request_id=&biz_id=102&utm_term=%E7%9B%B8%E6%9C%BA%E5%86%85%E5%8F%82%E8%AE%A1%E7%AE%97%20%E5%B7%B2%E7%9F%A5%E8%A7%86%E5%9C%BA%E8%A7%92&utm_medium=distribute.pc_search_result.none-task-blog-2~all~sobaiduweb~default-0-140603386.142^v102^control&spm=1018.2226.3001.4187)


**Compuation of ray direction:**

Why we use direction vector instead of point? Because we can't decide where the point actually is since points with the same direction and different depth can progject into the same pixel in the image.So we use direction vector to represent rays instead of point which will be sampled later. 

**Formula transforming pixel to direction vector:**
```
(u, v, 1) = K (xn, yn, 1) -> (xn, yn, 1) = K^-1(u, v, 1) -> since camera position is (0, 0, 0) in camera coordinate -> direction verctor = (xn, yn, 1) in camera coordinate -> transform into world coordinate
```

### Problems encontered
Probelm：np.broadcast_to function only return read-only view instead of writable numpy which will casue problem while transfroming into tensor

Solution：add .copy() to generate writable copy
```python
rays_o = np.broadcast_to(transform_matrix[:3,3], rays_d.shape).copy()
```
## Implement of Network class
Model file path: `src/models/nerf/network.py`

I have change the gpus from 1 to 0 in lego.yaml config file to make it fit for my computer and I have run the network successfully !

## Implement of Render class
Renderer file path: `src/models/nerf/renderer/volume_renderer.py`

### Overall Process of rendering
1) data preparation including flattening rays
2) generate coarse points by stratified_sample_points_from_rays function
3) generate rgb and density preded by coarse network
4) generate fine points by importance_sample_points function
5) combine both coarse and fine points as input of nerwork
6) generate rgb and density preded by fine network
7) generate rgb_values and depth_values by volume rendering method

### Implement of Sapmled points choice
**stratified sampling Steps:**
- (1)uniformally choose chin points in ray
- (2)randomly choose sampled points in each chin(train)
- (3)get different sampled points for each ray

**importance sampling Steps:**
- (1)compute weights of each chin of coarse points
- (2)compute cdf,pdf using weights
- (3)choose u in range(0,1) to generate sampled_points (train randomly,tesr lineraly)
- (4)get index of chin points both in cdf and depth corrporsending to position relationship between u and cdf
- (5)generate sampled points

### Problems encontered
Problem 1:I used boundouary of coarse points's depth instead of mids to generate sampled points

Problem 2:I didn't use different sampleing methods for different task like tarin or test.Because test task we shouldn't randomly choose sampled points which will cause lots of noisy points

