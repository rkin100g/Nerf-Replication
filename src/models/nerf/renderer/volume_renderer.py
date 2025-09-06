import numpy as np
import torch
from src.config import cfg
import torch.nn.functional as F


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
        self.sample_size = 64                            # 每批采样点处理数量
        self.rays_size = 102400                          # 每批光线处理数量


    def stratified_sample_points_from_rays(self, rays_o, rays_d, N_samples=64, t_n=2.0, t_f=6.0, perturb=True):
        """
        Inputs:
            rays_o: (N_rays, 3) torch tensor
            rays_d: (N_rays, 3) torch tensor
            t_n, t_f: decide the range of sampling according to dataset (float)
            perturb: 是否进行随机选点

        returns:
            t_sample: (N_rays, N_samples) depths along ray 采样点的深度与均匀采样得到的边界点深度区分
            sampled_points: (N_rays, N_samples, 3) sampled 3D points in world coords
        """
        # 数据准备
        device = self.device
        N_rays = rays_o.shape[0]  # 光线数量

        # 对ray进行均匀分段 
        t_linear = torch.linspace(t_n, t_f, N_samples, device=device)                # (N_samples,)
        t_sample = t_linear.unsqueeze(0).expand(N_rays, N_samples).clone()           # (N_rays, N_sampls)

        # 每段内进行随机选取点
        if perturb:
            # 计算每段期间重点，左边界，有边界
            mids = 0.5 * (t_linear[1:] + t_linear[:-1])                            # 每段中间点 (N_samples,)
            lower = torch.cat([t_linear[:1], mids], dim=0)                        # 每段左边界 (N_samples,)
            upper = torch.cat([mids, t_linear[-1:]], dim=0)                       # 每段右边界 (N_samples,)

            # 扩展至(N_rays, N_samples)
            lower = lower.unsqueeze(0).expand(N_rays, N_samples)
            upper = upper.unsqueeze(0).expand(N_rays, N_samples)

            # 计算每个ray的每段偏移量
            jitter = torch.rand(N_rays, N_samples, device = device)              # 每段偏移量 (N_rays, N_samples)
            t_sample = lower + (upper - lower) * jitter # (N_rays,N_samples)

        # 进而获得每个ray都不同的采样点
        sampled_points = rays_o.unsqueeze(1) + rays_d.unsqueeze(1) * t_sample.unsqueeze(2)  # (N_rays, N_samples, 3)
        
        return t_sample, sampled_points

    def weights_computation(self, density, t_sample):
        """
        Inputs:
            density: (N_rays, N_samples) 
            t_sample:  (N_rays, N_samples) 采样点的深度

        returns:
            weights: (N_rays, N_samples)
        """
        device = self.device

        # 计算概率密度分布
        # 计算ti+1 - ti,最后一个点没有下一个点，所以补一个很大的值作为间隔
        delta = t_sample[..., 1:] -t_sample[..., :-1]
        delta = torch.cat([delta, torch.tensor([1e10], device=device).expand(delta[..., :1].shape)], dim=-1)

        # 计算不透明度alpha (N_rays, N_samples)
        alpha = 1.0 - torch.exp(-density * delta)

        # 计算透射率
        temp = torch.cat([torch.ones(alpha.shape[0], 1, device=device), 1.0 - alpha + 1e-10], dim=-1)
        T = torch.cumprod(temp, dim=-1)[:, :-1]

        # 计算权重
        weights = T * alpha
        
        # 释放中间量，降低显存占用
        del delta, alpha, temp, T
        
        return weights
    
    def importance_sample_points(self, density_coarse, rays_o, rays_d, t_coarse, N_importance):
        """
        Inputs:
            density_coarse: (N_rays, N_samples) 粗采样点的密度
            rays_o: (N_rays, 3) torch tensor    
            rays_d: (N_rays, 3) torch tensor
            t_coarse:  (N_rays, N_samples)      粗采样点的深度

        returns:
            t_fine: (N_rays, N_importance) depths along ray 新采样点的深度
            fine_points: (N_rays, N_importance, 3) sampled 3D points in world coords
        """
        device = self.device
        # 计算权重
        weights = self.weights_computation(density=density_coarse, t_sample=t_coarse)   # (N_rays, N_samples)
        
        # 计算概率密度函数
        pdf = weights / (weights.sum(dim=-1, keepdim=True) + 1e-5)                  # (N_rays, N_samples)
        
        # 计算分布函数(累加和)
        cdf = torch.cumsum(pdf, dim=-1)                                             # (N_rays, N_samples)
        cdf = torch.cat([torch.zeros(cdf.shape[0], 1, device=device), cdf], dim=-1)                 

        # 均匀随机采样
        N_rays = rays_o.shape[0]
        u = torch.rand(N_rays, N_importance, device=device)

        # 逆变换获得index索引(u在cdf中对应右侧采样点的位置)
        sample_index = torch.searchsorted(cdf, u, right=True)
        sample_index = torch.clamp(sample_index, min=0, max=cdf.shape[-1] - 1)  # 上限为N_samples，而非N_samples-1

        # 确定采样点左右索引和左右采样点深度
        sample_index_below = torch.clamp(sample_index - 1, min=0, max=t_coarse.shape[-1] - 1)
        sample_index_above = torch.where(
            sample_index <= self.N_samples - 1,  # 条件：未超过t_coarse的最大索引
            sample_index,                  # 满足条件：用sample_index
            torch.tensor(self.N_samples - 1, device=device)  # 不满足条件：强制为t_coarse的最后一个索引
        )
        t_coarse_below = torch.gather(t_coarse, 1, sample_index_below)
        t_coarse_above = torch.gather(t_coarse, 1, sample_index_above)

        # 确定深度边界
        cdf_below = torch.gather(cdf, 1, sample_index_below)
        cdf_above = torch.gather(cdf, 1, sample_index_above)

        # 由线性比例来确定精确采样位置
        cdf_range = cdf_above - cdf_below
        cdf_range[cdf_range < 1e-5] = 1e-10
        t = (u - cdf_below) / cdf_range
        t_fine = t_coarse_below + t * (t_coarse_above - t_coarse_below)
        
        # 采样得到新的采样点
        fine_points = rays_o.unsqueeze(1) + rays_d.unsqueeze(1) * t_fine.unsqueeze(2)
        
        # 最后返回之前释放中间量，降低显存占用
        del weights, pdf, cdf, cdf_range

        return fine_points, t_fine
    

    def render(self, batch):
        """
        This function is responsible for rendering the output of the model, which includes the RGB values and the depth values.
        """
        #(按理说要分train和test进行不同的处理，或者有所区分，这里为了快速测试render部分没有问题，暂时不做区分)
        # Part 1:构造网络输入
        # 获得rays([batch_size,N_rays,N_samples])
        rays_o = batch["rays_o"]
        rays_d = batch["rays_d"]
        self.device = rays_o.device
        
        # 展平为([N_rays * batch_size,N_samples])
        B, N_rays, _ = rays_o.shape
        rays_o = rays_o.reshape(B * N_rays, 3)
        rays_d = rays_d.reshape(B * N_rays, 3)

        # ray进行粗采样，得到64个采样点位置xyz
        t_coarse, coarse_points = self.stratified_sample_points_from_rays(
            rays_o=rays_o, rays_d=rays_d, N_samples=self.N_samples, perturb=self.perturb
        )
        depth = t_coarse
        
        # Part 2:传入粗网络输出结果
        # 计算观测方向viewdirs(即ray_d单位向量)
        viewdirs = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)  # 单位向量 (N_rays, 3)

        print("Render-coarse_net processing")
        
        # 分块处理
        outputs = []
        for i in range(0, coarse_points.shape[0], self.rays_size):
            outputs.append(
                self.net.forward(coarse_points[i:i+self.rays_size,:],
                                viewdirs[i:i+self.rays_size], model="")
            )
        outputs = torch.cat(outputs, dim=0)

        print("Render-coarse_net finish processsing")
        

        # Part 3:由粗网络结果进行逆采样并传入细网络输出结果
        if self.N_importance > 0:
            # 获得粗网络结果
            density_coarse = outputs[...,3]
            
            # 进行relu，限制范围为非负
            density_coarse_relu = F.relu(density_coarse)
            
            # 重要性采样
            fine_points, t_fine = self.importance_sample_points(density_coarse_relu, rays_o, rays_d, t_coarse, self.N_importance)

            # 合并粗采样点和细采样点
            sampled_points = torch.cat([coarse_points, fine_points],dim=1)
            depth = torch.cat([t_coarse, t_fine], dim=1)
            
            # 用完就释放，减少显存占用
            del coarse_points, fine_points
            del t_coarse, t_fine

            # 排序，便于计算权重
            depth, indices = torch.sort(depth, dim=-1)  # (N_rays, N_samples_total)
            sampled_points_sorted = torch.gather(
                sampled_points, 1, indices.unsqueeze(-1).expand(-1, -1, 3)
            )
            
            # 用完就释放，减少显存占用
            del sampled_points, indices
            
            print("Render-importance sampleing fished and fine_net processing")

            #细网络分块处理
            outputs = []
            for i in range(0,sampled_points_sorted.shape[0],self.rays_size):
                outputs_chunk = []
                for j in range(0, sampled_points_sorted.shape[1], self.sample_size):
                    outputs_chunk.append(
                        self.net.forward(sampled_points_sorted[i:i+self.rays_size,j:j+self.sample_size],
                                        viewdirs[i:i+self.rays_size], model="fine")
                    )
                outputs_chunk = torch.cat(outputs_chunk, dim=1)
                outputs.append(outputs_chunk)
            outputs = torch.cat(outputs, dim=0)
            
            print("Render-fine_net finish processing")

        # Part5:体渲染得到渲染图
        # 从ouputs中获得网络输出的rgb和density
        rgb = outputs[..., :3]                                               # (N_rays, N_samples, 3)
        density = outputs[...,3]

        # 对density进行进行relu，限制范围为非负，对rgb范围限制在0-1
        rgb = F.sigmoid(rgb)
        density_relu = F.relu(density)

        # 计算权重
        weights = self.weights_computation(density=density_relu, t_sample=depth)   # (N_rays, N_samples)

        # 求和得到每个像素的rgb与深度值
        rgb_values = torch.sum(weights.unsqueeze(-1) * rgb, dim=1)  # (N_rays, 3)
        depth_values = torch.sum(weights * depth, dim=1)          # (N_rays,)
        
        # 对空区域（权重和 < 0.01）的depth设为far
        weights_sum = torch.sum(weights, dim=1)
        #depth = torch.where(weights_sum < 0.01, torch.tensor(0.6, device=self.device), depth)

        # 白色背景处理
        if self.white_bkgd:
            rgb_values = rgb_values + (1.0 - weights.sum(dim=-1, keepdim=True))

        return rgb_values, depth_values

        


    