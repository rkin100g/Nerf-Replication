import torch.utils.data as data
import torch
import numpy as np
from src.config import cfg


class Dataset(data.Dataset):
    def __init__(self, **kwargs):
        """
        Description:
            __init__ 函数负责从磁盘中 load 指定格式的文件，计算并存储为特定形式

        Input:
            @kwargs: 读取的参数
            由实例化过程可知传入参数为:cfg.test/train_dataset(没有val,设置默认值)
            包括data_root,split,input_ratio,cams,H,W
        Output:
            None
        """
        super(Dataset, self).__init__()

        import os
        import json
        import imageio
        import cv2
        # 参数读取
        self.data_root = kwargs.get("data_root", "data/nerf_synthetic")
        self.data_root = os.path.join(self.data_root,"lego")
        self.split = kwargs.get("split", "val")            # 默认为val部分
        self.H = kwargs.get("H", 800)                      # 默认高度
        self.W = kwargs.get("W", 800)                      # 默认宽度
        self.input_ratio = kwargs.get("input_ratio", 1)  # 默认下采样

        # 读取json文件
        json_path = os.path.join(self.data_root, f"transforms_{self.split}.json")
        with open(json_path, "r") as f:
            meta = json.load(f)

        # 相机视场角（用于计算内参）
        self.camera_angle_x = meta["camera_angle_x"]

        # 建立每帧数据与index的映射
        self.data = []

        # 图像下采样
        if self.input_ratio != 1.0:
            self.H, self.W = int(self.H * self.input_ratio), int(self.W * self.input_ratio)

        for frame in meta["frames"]:
            # 拼接图像路径与图像下采样
            img_path = os.path.join(self.data_root, frame["file_path"]+".png")
            image = np.array(imageio.imread(img_path), dtype=np.float32, copy=True)
            image = image[..., :3]  # 只保留RGB通道，丢弃Alpha
            image = cv2.resize(image, (self.W, self.H), interpolation=cv2.INTER_LINEAR)

            if self.split == "test":
                depth_path = os.path.join(self.data_root, frame["file_path"] + "_depth_0001.png")
                normal_path = os.path.join(self.data_root, frame["file_path"] + "_normal_0001.png")
                depth_img = np.array(imageio.imread(depth_path), dtype=np.float32, copy=True)
                normal_img = np.array(imageio.imread(normal_path), dtype=np.float32, copy=True)

                # 深度图和法线图下采样
                # 深度图下采样用最近邻插值
                depth_img = cv2.resize(depth_img, (self.W, self.H), interpolation=cv2.INTER_NEAREST)            # 深度图采样方式修改
                normal_img = cv2.resize(normal_img, (self.W, self.H), interpolation=cv2.INTER_LINEAR)
                
                # 保存每帧数据
                frame_data={
                    "image":image,                                       # 对应图像
                    "transform_matrix":frame["transform_matrix"],        # 变换矩阵
                    "depth":depth_img,                                   # 深度图
                    "normal":normal_img,                                 # 法线图
                }
            
            else: #train / val
                frame_data={
                    "image":image,                     # 对应图像
                    "transform_matrix":frame["transform_matrix"],        # 变换矩阵
                }

            self.data.append(frame_data)
        

    def __getitem__(self, index):
        """
        Description:
            __getitem__ 函数负责在运行时提供给网络一次训练需要的输入，以及 ground truth 的输出
        对 NeRF 来说，分别是 1024 条光线以及 1024 个 RGB值

        Input:
            @index: 图像下标, 范围为 [0, len-1]
        Output:
            @ret: 包含所需数据的字典
        """
        from math import tan
        # 获得该帧数据
        frame = self.data[index]
        image = frame["image"]
        transform_matrix = np.array(frame["transform_matrix"])

        # 获得相机内参：焦距，中心点坐标
        f = self.W / (2 * tan(self.camera_angle_x / 2))
        cx, cy = self.W / 2, self.H / 2
        
        # 像素采样
        if self.split == "train":
            # 随机采样1024个像素
            idx = np.random.choice(self.H * self.W, 1024, replace=False)
            u = idx % self.W
            v = idx // self.W
            sampled_pixels = np.stack([u, v], axis=-1)
        else :
            # 使用所有像素
            us, vs = np.meshgrid(np.arange(self.W), np.arange(self.H))
            sampled_pixels = np.stack([us.ravel(), vs.ravel()], -1)

        # 记录各自位置上的RGB颜色  y,x                               归一化
        colors = image[sampled_pixels[:, 1], sampled_pixels[:, 0]] / 255.0

        # 获得每个像素点对应的相机坐标系下的方向向量->获得光线方向ray_d
        u, v = sampled_pixels[:, 0], sampled_pixels[:, 1]
        dirs = np.stack([(u - cx) / f, -(v - cy) / f, -np.ones_like(u)], axis=-1)

        # 转到世界坐标系
        rays_d = (transform_matrix[:3,:3] @ dirs.T).T
        rays_d = rays_d / np.linalg.norm(rays_d, axis=-1, keepdims=True)  # 单位化
        rays_o = np.broadcast_to(transform_matrix[:3,3], rays_d.shape).copy()

        if self.split == "test":
            # 获得深度和法线信息
            depth_img = frame["depth"]
            normal_img = frame["normal"]
            depth = depth_img[sampled_pixels[:, 1], sampled_pixels[:, 0]] 
            normal = normal_img[sampled_pixels[:, 1], sampled_pixels[:, 0]] / 255.0

            """
            # 将深度图中的相机坐标系下的深度信息转到世界坐标系下
            # 相机坐标系下的点：(dirs * depth) → 方向向量乘以深度
            cam_coords = dirs * depth[..., None]  # (N, 3)，相机坐标系下的点
            # 2. 转换到世界坐标系：世界点 = R * 相机点 + T
            world_coords = (transform_matrix[:3,:3] @ cam_coords.T).T + transform_matrix[:3,3]  # (N, 3)
            # 3. 世界坐标系深度 = 世界点到光线起点（相机原点）的距离
            depth_world = np.linalg.norm(world_coords - rays_o, axis=-1)  # (N,)
            # 用世界坐标系深度替换原始深度
            depth = depth_world
            """

            data_dict ={
                "colors":torch.from_numpy(colors).float(),
                "rays_o":torch.from_numpy(rays_o).float(),
                "rays_d":torch.from_numpy(rays_d).float(),
                "depth":torch.from_numpy(depth).float(),
                "normal":torch.from_numpy(normal).float(),
            }
        
        else: # train / val
            # 构造data字典输出
            data_dict ={
                "colors":torch.from_numpy(colors).float(),
                "rays_o":torch.from_numpy(rays_o).float(),
                "rays_d":torch.from_numpy(rays_d).float(),
            }

        return data_dict

    def __len__(self):
        """
        Description:
            __len__ 函数返回训练或者测试的数量

        Input:
            None
        Output:
            @len: 训练或者测试的数量
        """
        return len(self.data)
