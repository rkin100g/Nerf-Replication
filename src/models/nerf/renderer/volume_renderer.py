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
        self.rays_size = 160000                          # 每批光线处理数量
        self.task = getattr(cfg, "task", "test")
        self.perturb = self.perturb if self.task == "train" else False   # 如果是测试阶段，就不进行随机采样
        self.fast_sampling = getattr(cfg, "fast_sampling", False)
        self.weights_threshold = getattr(cfg, "weights_threshold", 0.01)


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

        # 每段内进行随机选取点(train),均匀采样(test)
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
        delta = torch.cat([delta, 1e10 * torch.ones_like(delta[..., :1])], dim=-1)

        # 计算不透明度alpha (N_rays, N_samples)
        alpha = 1.0 - torch.exp(-density * delta)

        # 计算透射率
        temp = torch.cat([
            torch.ones(alpha.shape[0], 1, device=device),
            torch.clamp(1.0 - alpha, min=1e-10, max=1.0)  # 限制在[1e-10, 1.0]，避免T>1
        ], dim=-1)
        T = torch.cumprod(temp, dim=-1)[:, :-1]

        # 计算权重
        weights = T * alpha
        
        return weights

    def importance_sample_points(self, density_coarse, rays_o, rays_d, t_coarse, N_importance, eps = 1e-5):
        """
        Inputs:
            density_coarse: (N_rays, N_samples) 粗采样点的密度
            rays_o: (N_rays, 3) torch tensor    
            rays_d: (N_rays, 3) torch tensor
            t_coarse:  (N_rays, N_samples)      粗采样点的深度
            eps:偏置避免除以0

        returns:
            t_fine: (N_rays, N_importance) depths along ray 新采样点的深度
            fine_points: (N_rays, N_importance, 3) sampled 3D points in world coords
        """

        device = self.device
        
        # Step 1:计算归一化权重
        weights = self.weights_computation(density=density_coarse, t_sample=t_coarse)
        weights = weights[..., 1:-1]  # 取中间N_samples-1个间隔的权重
        print(f"weights: {np.array2string(weights[40510,:].cpu().detach().numpy(), precision=5)}")
        
        weights = weights + eps  # 防止数值不稳定
        pdf = weights / torch.sum(weights, -1, keepdim=True)  # [N_rays, N_samples-1]
        cdf = torch.cumsum(pdf, -1)  # [N_rays, N_samples-1]
        cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)  # [N_rays, N_samples]

        # Step 2:生成采样点[训练时随机,测试时确定]
        if self.task == "train":               # 训练模式,随机采样避免过拟合
            u = torch.rand(list(cdf.shape[:-1]) + [N_importance], device=device)                 # 随机均匀采样
        else:                                   # 测试模式,确保渲染可重复,减少噪点存在
            u = torch.linspace(0., 1., steps=N_importance, device=device)  # 线性均匀采样
            u = u.expand(list(cdf.shape[:-1]) + [N_importance])  # 扩展到与射线数量匹配
        
        # Step 3:线性插值找对应cdf采样位置(原方案为获得index后各自在cdf和depth里找左右索引，后发现nerf代码直接构造了左右边界的代码组合更简洁)
        u = u.contiguous()
        inds = torch.searchsorted(cdf, u, right=True)                        # 找到u在CDF中的位置
        print("inds:",inds[40510,:])
        below = torch.clamp(inds - 1, 0, cdf.shape[-1]-1)                    # 确定左边界
        above = torch.clamp(inds, 0, cdf.shape[-1]-1)                        # 确定右边界
        inds_g = torch.stack([below, above], -1)  # [N_rays, N_importance, 2]# 构造采样区间

        # Step 4:计算区间中点，因为区间的权重可以近似看成是区间中点的权重，由此一一对应更加合适
        bins = 0.5 * (t_coarse[..., 1:] + t_coarse[..., :-1])  # 此时 bins 形状为 (N_rays, N_samples-1)

        # Step 5:从bins中插值出采样点
        # 统一形状,用于拓展出细采样点的左右边界
        matched_shape = [inds_g.shape[0], inds_g.shape[1], bins.shape[-1]]      # (N_rays,N_importance,N_samples-1) 
        
        # 获取每个采样点对应cdf上下边界
        cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
        
        # 提取每个采样点的深度
        bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)
        
        # 计算CDF区间长度,同时避免除以0
        denom = cdf_g[..., 1] - cdf_g[..., 0]
        denom = torch.where(denom < eps, torch.ones_like(denom), denom)  # 避免除零
        
        # 获取比例，采样点深度
        t = (u - cdf_g[..., 0]) / denom
        t_fine = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])
        fine_points = rays_o.unsqueeze(1) + rays_d.unsqueeze(1) * t_fine.unsqueeze(2)

        print("t_fine:",t_fine[40510,:])
        fine_valid_mask = None
        return fine_points, t_fine, fine_valid_mask

    def fine_sample_points(self, density_coarse, rays_o, rays_d, t_coarse, N_importance, N_samples, weights_threshold, eps=1e-5):
        """
        Inputs:
            density_coarse: (N_rays, N_samples) 粗采样点的密度
            rays_o: (N_rays, 3) torch tensor    
            rays_d: (N_rays, 3) torch tensor
            t_coarse:  (N_rays, N_samples)      粗采样点的深度
            eps:偏置避免除以0

        returns:
            t_fine: (N_rays, N_importance) depths along ray 新采样点的深度
            fine_points: (N_rays, N_importance, 3) sampled 3D points in world coords
        """
        device = self.device
        valid_mask = None
        
        # Part 1:判断空区间并记录下是否为空区间，判断标准为weights小于一定阈值(如果开启ESS)
        # 计算权重并剔除第一个和最后一个人为补的权重
        weights = self.weights_computation(density=density_coarse, t_sample=t_coarse)
        weights = weights[..., 1:-1]        # (N_rays, N_samples -2)
        
        # 判断是否为空区间并记录
        if self.fast_sampling == True:
            empty_bins = (weights < weights_threshold)   # 直接返回布尔张量，True表示空区间,(N_rays, N_samples -2)

        # Part 2:筛选出有效区间
        # Step 1:计算cdf
        weights = weights + eps  
        pdf = weights / torch.sum(weights, -1, keepdim=True)        # [N_rays, N_samples-1]
        cdf = torch.cumsum(pdf, -1)                                 # [N_rays, N_samples-1]
        cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)  # [N_rays, N_samples]

        # Step 2:生成采样点
        if self.task == "train":               
            u = torch.rand(list(cdf.shape[:-1]) + [N_importance], device=device)                 
        else:                                   
            u = torch.linspace(0., 1., steps=N_importance, device=device)  
            u = u.expand(list(cdf.shape[:-1]) + [N_importance])  
        
        # Step 3:线性插值找对应cdf采样位置
        u = u.contiguous()
        inds = torch.searchsorted(cdf, u, right=True)                        
        print("inds:",inds[40200,:])

        below = torch.clamp(inds - 1, 0, N_samples -3)             # weights是N_samples - 2，最大索引为N_samples -3         
        above = torch.clamp(inds, 0, N_samples -3)     
        inds_g = torch.stack([below, above], -1)                   # [N_rays, N_importance, 2]

        # Step 4:筛选空区间得到掩码
        if self.fast_sampling == True :
            ray_indices = torch.arange(empty_bins.shape[0], device=device).unsqueeze(1).repeat(1, N_importance)

            # 使用二维索引直接访问，确保维度匹配
            below_empty = empty_bins[ray_indices, below]  # [N_rays, N_importance-2]
            above_empty = empty_bins[ray_indices, above]  # [N_rays, N_importance-2]

            both_empty = below_empty & above_empty         # [N_rays, N_importance-2]
            valid_mask = ~both_empty                       # [N_rays, N_importance-2]，确保二维

        # Part 3:采样
        # Step 1:计算区间中点
        bins = 0.5 * (t_coarse[..., 1:] + t_coarse[..., :-1])       # [N_rays, N_samples-1]

        # Step 2:统一形状,用于拓展出细采样点的左右边界
        matched_shape = [inds_g.shape[0], inds_g.shape[1], bins.shape[-1]]      # (N_rays,N_importance,N_samples-1) 
        
        # Step 3：获取每个采样点对应cdf上下边界和深度
        cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
        bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)
        
        # Step 4::计算CDF区间长度,同时避免除以0
        denom = cdf_g[..., 1] - cdf_g[..., 0]
        denom = torch.where(denom < eps, torch.ones_like(denom), denom)  # 避免除零
        
        # Step 5:获取比例，采样点深度
        t = (u - cdf_g[..., 0]) / denom
        t_fine = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

        # Step 7:采样点
        fine_points = rays_o.unsqueeze(1) + rays_d.unsqueeze(1) * t_fine.unsqueeze(2)
        
        # Step 4：实时跟踪通透度，达到阈值时停止后续采样 
        #total_valid = torch.sum(valid_mask).item()  # 转成Python数值，方便打印  
        #print("total points num:",total_valid)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    
        return fine_points, t_fine, valid_mask
    
    def safe_slice(self, valid_mask, i, j, rays_size, sample_size):
        """
        splic operation safely:if mask is None then return None else return slice of valid
        Inputs:
            mask: 可能为None的掩码数组
            i, j: 切片起始坐标
            rays_size, sample_size: 切片的高度和宽度
        Outputs:
            splice result
        """
        if valid_mask == None:
            return None
        else:
            return valid_mask[i:i+rays_size,j:j+sample_size]
    
    def render(self, batch):
        """
        This function is responsible for rendering the output of the model, which includes the RGB values and the depth values.
        """
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
                                viewdirs[i:i+self.rays_size],valid_mask=None, model="")
            )
        outputs = torch.cat(outputs, dim=0)

        print("Render-coarse_net finish processsing")
        

        # Part 3:由粗网络结果进行逆采样并传入细网络输出结果
        if self.N_importance > 0:
            # 获得粗网络结果
            density_coarse = outputs[...,3]
    
            # 进行relu，限制范围为非负
            density_coarse_relu = F.relu(density_coarse)
            
            # 统一的细采样
            fine_points, t_fine, fine_valid_mask = self.fine_sample_points(
                density_coarse=density_coarse_relu,
                rays_o=rays_o, rays_d=rays_d, 
                t_coarse=t_coarse, 
                N_importance = self.N_importance, N_samples = self.N_samples,
                weights_threshold = self.weights_threshold)

            # 合并粗采样点和细采样点
            sampled_points = torch.cat([coarse_points, fine_points],dim=1)
            depth = torch.cat([t_coarse, t_fine], dim=1)
            
            # 用完就释放，减少显存占用
            del coarse_points, fine_points
            del t_coarse, t_fine

            # 排序采样点和深度，便于计算权重
            depth, indices = torch.sort(depth, dim=-1)  # (N_rays, N_samples_total)
            sampled_points_sorted = torch.gather(
                sampled_points, 1, indices.unsqueeze(-1).expand(-1, -1, 3)
            )
            
            # 掩码合并与排序
            if self.fast_sampling == True:
                # 构建粗采样点掩码，全部为True
                coarse_valid_mask = torch.ones(size=(B * N_rays, self.N_samples),dtype=torch.bool,device=self.device)

                # 合并掩码
                valid_mask = torch.cat([coarse_valid_mask, fine_valid_mask],dim=1)
            
                # 排序掩码，一一对应
                valid_mask_sorted = torch.gather(
                    valid_mask, 1, indices
                )
            else:
                valid_mask_sorted = None
            
            # 用完就释放，减少显存占用
            del sampled_points, indices

            print("Render-importance sampleing fished and fine_net processing")
            
            #细网络分块处理
            outputs = []
            for i in range(0,sampled_points_sorted.shape[0],self.rays_size):
                outputs_chunk = []
                for j in range(0, sampled_points_sorted.shape[1], self.sample_size):
                    # 构建切片(可能为None)
                    valid_mask_slice = self.safe_slice(valid_mask_sorted, i, j, self.rays_size, self.sample_size)
                    outputs_chunk.append(
                        self.net.forward(sampled_points_sorted[i:i+self.rays_size,j:j+self.sample_size],
                                        viewdirs[i:i+self.rays_size], valid_mask_slice, model="fine")
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

        # 白色背景处理
        if self.white_bkgd:
            rgb_values = rgb_values + (1.0 - weights.sum(dim=-1, keepdim=True))

        return rgb_values, depth_values

        


    