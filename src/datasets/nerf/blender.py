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
        Output:
            None
        """
        super(Dataset, self).__init__()

        import os
        import json
        import imageio

        # 数据集目录
        data_root = kwargs["data_root"]

        # 读取json文件
        with open("transforms_train.json", "r") as f:
            meta = json.load(f)

        # 相机视场角（用于计算内参）
        self.camera_angle_x = meta["camera_angle_x"]

        # 建立每帧数据与index的映射
        self.data = []

        for frame in meta["frames"]:
            # 拼接图像路径
            img_path = os.path.join(data_root, frame["file_path"]+".png")
            
            # 保存每帧数据
            frame_data={
                "image":imageio.imread(img_path),                     # 对应图像
                "rotation":frame["rotation"],                       # 旋转角度
                "transform_matrix":frame["transform_matrix"]        # 变换矩阵
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

        # 计算相机焦距
        H, W = image.shape[:2]
        f = W / (2 * tan(self.camera_angle_x / 2))

        # 获得相机内参矩阵
        cx, cy = W / 2, H / 2
        K = np.array([
            [f, 0, cx],
            [0, f, cy],
            [0, 0, 1]
        ])
        
        # 随机采样1024个像素
        idx = np.random.choice(H*W, 1024, replace=False)
        u = idx % W
        v = idx // W
        sampled_pixels = np.stack([u, v], axis=-1)

        # 记录各自位置上的RGB颜色  y,x
        colors = image[sampled_pixels[:, 1], sampled_pixels[:, 0]]

        # 获得每个像素点对应的相机坐标系下的方向向量->获得光线方向ray_d
        u, v = sampled_pixels[:, 0], sampled_pixels[:, 1]
        dirs = np.stack([(u - cx) / f, -(v - cy) / f, -np.ones_like(u)], axis=-1)

        # 转到世界坐标系
        rays_d = (transform_matrix[:3,:3] @ dirs.T).T
        rays_o = np.broadcast_to(transform_matrix[:3,3], rays_d.shape)

        # 构造data字典输出
        data_dict ={
            "colors":colors,
            "rays_o":rays_o,
            "rays_d":rays_d
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
