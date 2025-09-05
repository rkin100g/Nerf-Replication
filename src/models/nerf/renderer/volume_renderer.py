import numpy as np
import torch
from src.config import cfg


class Renderer:
    def __init__(self, net):
        """
        This function is responsible for defining the rendering parameters, including the number of samples, the step size, and the background color.
        """
        self.net = net
        self.N_samples = getattr(cfg, "N_samples", 64)
        self.chunk_size = getattr(cfg, "chunk_size", 1024)
        self.white_bkgd = getattr(cfg, "white_bkgd", True)
        self.N_importance = getattr(cfg, "N_importance", 128)
        self.perturb = getattr(cfg, "perturb", True)


    def stratified_sample_points_from_rays(self, rays_o, rays_d, N_samples=64, t_n=2.0, t_f=6.0, perturb=True):
        """
        Inputs:
            rays_o: (N_rays, 3) torch tensor
            rays_d: (N_rays, 3) torch tensor
            t_n, t_f: decide the range of sampling according to dataset (float)

        returns:
            t_vals: (N_rays, N_samples) depths along ray
            sampled_points:    (N_rays, N_samples, 3) sampled 3D points in world coords
        """
        # 数据准备
        device = rays_o.device
        N_rays = rays_o.shape[0]  # N_rays = batch_size * N_rays

        # 对ray进行均匀分段 
        t_lin = torch.linspace(t_n, t_f, N_samples, device=device)           # (N_samples,)
        t_vals = t_lin.unsqueeze(0).expand(N_rays, N_samples).clone()        # (N_rays, N_sampls)

        # 每段内进行随机选取点
        if perturb:
            # 计算每段期间重点，左边界，有边界
            mids = 0.5 * (t_lin[1:] + t_lin[:-1])                            # 每段中间点 (N_samples,)
            lower = torch.cat([t_lin[:1], mids], dim=0)                        # 每段左边界 (N_samples,)
            upper = torch.cat([mids, t_lin[-1:]], dim=0)                       # 每段右边界 (N_samples,)

            # 扩展至(N_rays, N_samples)
            lower = lower.unsqueeze(0).expand(N_rays, N_samples)
            upper = upper.unsqueeze(0).expand(N_rays, N_samples)

            # 计算每个ray的每段偏移量
            jitter = torch.rand(N_rays, N_samples, device = device)              # 每段偏移量 (N_rays, N_samples)
            t_vals = lower + (upper - lower) * jitter # (N_rays,N_samples)

        # 进而获得每个ray都不同的采样点
        sampled_points = rays_o.unsqueeze(1) + rays_d.unsqueeze(1) * t_vals.unsqueeze(2)  # (N_rays, N_samples, 3)
        return t_vals, sampled_points

    def render(self, batch):
        """
        This function is responsible for rendering the output of the model, which includes the RGB values and the depth values.
        """
        # Part 1:构造网络输入
        # chunk_size 构造（后续完成
        # 获得rays([batch_size,N_rays,N_samples])并展平为([N_rays * batch_size,N_samples])
        rays_o = batch["rays_o"]
        rays_d = batch["rays_d"]
        B, N_rays, _ = rays_o.shape
        rays_o = rays_o.reshape(B * N_rays, 3)
        rays_d = rays_d.reshape(B * N_rays, 3)

        # 每个ray上进行粗采样，得到64个采样点位置xyz
        t_vals, sampled_points = self.stratified_sample_points_from_rays(rays_o=rays_o, rays_d=rays_d, N_samples=self.N_samples, perturb=self.perturb)
        """
        print("t_vals shape:", t_vals.shape)
        print("t_vals :", t_vals)
        print("sampled_points shape:",sampled_points.shape)
        print("sampled_points:",sampled_points)
        """
        # 对采样点位置进行位置编码

        # 组合位置与位置编码构建好网络输入

       
        # Part 2:粗网络输出rgb和不透明度
        # 传入网络第一部分MLP

        # 再增加位置编码信息（残差结构）

        # 再传入网络第二部分MLP，输出不透明度

        # 增加观测角度与位置编码，构造第三部分MLP输入

        # 经过FC 和 linear层输出采样点位置对应的rgb颜色

        
        # Part 3:细网络输出准确结果
        # 由粗网络结果逆采样得到128个采样点

        # 重复Part2跑细网络


        # Part4:体渲染得到渲染图
        # 对各个ray上采样点的颜色和不透明度进行离散积分求和

        # 想办法生成深度图


    