import numpy as np
import torch
from src.config import cfg
import torch.nn.functional as F
import time


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
        self.weights_threshold = getattr(cfg, "weights_threshold", 0.25)


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
        
        return T,weights

    def fine_sample_points(self, density_coarse, rays_o, rays_d, t_coarse, N_importance, N_samples, weights_threshold, eps=1e-5,ert_threshold=0.45):
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
        N_rays = rays_o.shape[0]
        
        # Part 1：判断空射线
        density_sum = density_coarse.sum(dim=-1)  # (N_rays,)  每条射线的总密度
        empty_ray_mask = density_sum < 1e-3  # (N_rays,)  True表示空射线

        # 区分物体射线和背景射线
        density_max = density_coarse.max(dim=-1).values  # (N_rays,)  每条射线的最大密度
        object_ray_threshold = 0.5
        object_ray_mask = density_max > object_ray_threshold  # (N_rays,) True=物体射线
        background_ray_mask = (~object_ray_mask) & (~empty_ray_mask)  # (N_rays,) True=非空背景射线

        # Part 1:判断空区间并记录下是否为空区间，判断标准为weights小于一定阈值(如果开启ESS)
        # 计算权重并剔除第一个和最后一个人为补的权重
        T,weights = self.weights_computation(density=density_coarse, t_sample=t_coarse)
        weights = weights[..., 1:-1]        # (N_rays, N_samples -2)
        T = T[..., 1:-1]  # 同步剔除首尾，与weights维度一致：(N_rays, N_samples-2)
        
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

        below = torch.clamp(inds - 1, 0, N_samples -3)             # weights是N_samples - 2，最大索引为N_samples -3         
        above = torch.clamp(inds, 0, N_samples -3)     
        inds_g = torch.stack([below, above], -1)                   # [N_rays, N_importance, 2]

        # Step 4:筛选空区间得到掩码
        if self.fast_sampling == True :
            # Part 1:掩码构建
            # (1) ERT 无效掩码
            ert_mask_base = T < ert_threshold  # [N_rays, N_samples-2]，True表示该点T过低
            ert_mask_padded = torch.cat([
                torch.zeros_like(ert_mask_base[:, :1]),  # 首列补False 
                ert_mask_base
            ], dim=1)  # [N_rays, N_samples-1]
            ert_empty_bins = torch.cummax(ert_mask_padded, dim=1)[0][:, 1:]  # [N_rays, N_samples-2] cummax会让后续所有位置保持True
            
            row_idx = torch.arange(ert_empty_bins.shape[0], device=device).unsqueeze(1).expand_as(below)  # 从粗区间掩码中，为每个细采样点取对应的ERT标记：(N_rays, N_importance)
            ert_non_valid = ert_empty_bins[row_idx, below]   # 现在both_empty和ert_mask_for_fine维度一致，可合并
            ert_valid = ~ert_non_valid  # ERT单独有效：True=有效

            # (2) ESS 无效掩码
            ray_indices = torch.arange(empty_bins.shape[0], device=device).unsqueeze(1).repeat(1, N_importance)
            below_empty = empty_bins[ray_indices, below]  # [N_rays, N_importance-2]
            above_empty = empty_bins[ray_indices, above]  # [N_rays, N_importance-2]

            ess_non_valid_object = below_empty & above_empty  # 物体射线的ESS无效掩码
            # ② 背景射线：宽松判断（below或above空就无效，保证过滤效率）
            ess_non_valid_background = below_empty | above_empty  # 背景射线的ESS无效掩码
            # ③ 合并：用object_ray_mask选择对应逻辑（广播到采样点维度）
            object_mask_expand = object_ray_mask.unsqueeze(1).expand_as(ess_non_valid_object)  # [N_rays, N_importance]
            ess_non_valid = torch.where(
                object_mask_expand,  # 条件：是否是物体射线
                ess_non_valid_object,  # 是：用严格逻辑
                ess_non_valid_background  # 否：用宽松逻辑
            )
            ess_valid = ~ess_non_valid

            # (3) 合并掩码
            non_valid_mask = ess_non_valid | ert_non_valid  # 合并无效掩码
            valid_mask = ~non_valid_mask 
            valid_mask[empty_ray_mask] = False              # 空射线全设为False
        

            # Part 2:数量计算
            # (1) 基础信息
            total_points = N_rays * N_importance               # 细采样总点数
            print("="*49 + " ESS/ERT 过滤效果调试 " + "="*49)
            print(f"1. 基础信息：")
            print(f"   - 总射线数量：{N_rays}")
            print(f"   - 总细采样点数量：{total_points}")
            print(f"   - ESS 权重阈值：{weights_threshold}, ERT 透射度阈值：{ert_threshold}")


            # (2) 空射线过滤效果
            empty_ray_count = empty_ray_mask.sum().item()
            valid_ray_count = N_rays -  empty_ray_count  # 有效射线数
            empty_only_valid_count = empty_ray_count * N_importance
            print(f"\n2. 空射线过滤效果：")
            print(f"   - 空射线数量：{empty_ray_count}(条)")
            print(f"   - 有效射线数（排除空射线）：{valid_ray_count}（条）")
            print(f"   - 空射线滤过采样点数量：{empty_only_valid_count}（{empty_only_valid_count/total_points*100:.2f}%）")

            # (3) ESS单独过滤效果
            ess_only_valid = ess_valid & valid_mask
            ess_only_valid_count = ess_only_valid.sum().item()
            ess_reduce_count = total_points - ess_only_valid_count - empty_only_valid_count # ESS单独减少的点数
            print(f"\n3. ESS 单独效果：")
            print(f"   - ESS 单独有效点数量：{ess_only_valid_count}")
            print(f"   - ESS 单独减少点数：{ess_reduce_count}（{ess_reduce_count/total_points*100:.2f}%）")
            # 额外打印 ESS 权重低于阈值的粗区间占比（判断ESS是否正常工作）
            ess_coarse_empty_ratio = empty_bins.sum().item() / (empty_bins.shape[0] * empty_bins.shape[1])
            print(f"   - 粗区间中 ESS 判断为空的比例：{ess_coarse_empty_ratio*100:.2f}%")

            # 3.2 ERT 单独过滤后的有效点：ERT有效 + 有效射线
            ert_only_valid = ert_valid & valid_mask
            ert_only_valid_count = ert_only_valid.sum().item()
            ert_reduce_count = total_points - ert_only_valid_count - empty_only_valid_count # ERT单独减少的点数
            print(f"\n4. ERT 单独效果：")
            print(f"   - ERT 单独有效点数量：{ert_only_valid_count}")
            print(f"   - ERT 单独减少点数：{ert_reduce_count}（{ert_reduce_count/total_points*100:.2f}%）")
            # 额外打印 ERT 透射度低于阈值的点占比（判断ERT是否正常工作）
            ert_coarse_low_ratio = ert_mask_base.sum().item() / (ert_mask_base.shape[0] * ert_mask_base.shape[1])
            print(f"   - 粗区间中 ERT 判断为透射度过低的比例：{ert_coarse_low_ratio*100:.2f}%")
            print(f"   - 粗采样透射度 T 平均值：{T.mean().item():.6f}（低于 {ert_threshold} 才会过滤）")

            # 3.3 ESS+ERT 共同过滤后的有效点（最终结果）  
            final_valid_count = valid_mask.sum().item()
            total_reduce_count = total_points - final_valid_count  # 总共减少的点数
            common_reduce_count = total_reduce_count - empty_only_valid_count  # 正确公式
            print(f"\n5. ESS+ERT 共同效果：")
            print(f"   - 总共减少点数：{total_reduce_count}（{total_reduce_count/total_points*100:.2f}%）")
            print(f"   - ESS,ERT共同减少点数：{common_reduce_count}（{common_reduce_count/total_points*100:.2f}%）")
            print("="*118)


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

                #sampled_points_sorted = fine_points
                #depth = t_fine
            else:
                valid_mask_sorted = None
            
            # 用完就释放，减少显存占用
            del sampled_points, indices,coarse_points, fine_points,t_coarse, t_fine

            print("Render-importance sampleing fished and fine_net processing")
            
            #细网络分块处理
            torch.cuda.synchronize()
            start_time = time.perf_counter()

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

            #outputs = self.net.forward(sampled_points_sorted, viewdirs, valid_mask_sorted, depth, model="fine")

            # 记录结束时间
            end_time = time.perf_counter()

            # 计算耗时（秒）
            torch.cuda.synchronize()
            elapsed_time = end_time - start_time
            
            # 打印阶段与耗时结果（保留6位小数，更易读）
            print("Render-fine_net finish processing")
            print(f"细网络输出过程耗时:{elapsed_time:.6f} 秒")

        # Part5:体渲染得到渲染图
        # 从ouputs中获得网络输出的rgb和density
        rgb = outputs[..., :3]                                               # (N_rays, N_samples, 3)
        density = outputs[...,3]

        # 对density进行进行relu，限制范围为非负，对rgb范围限制在0-1
        rgb = F.sigmoid(rgb)
        density_relu = F.relu(density)

        # 计算权重
        T,weights = self.weights_computation(density=density_relu, t_sample=depth)   # (N_rays, N_samples)

        # 求和得到每个像素的rgb与深度值
        rgb_values = torch.sum(weights.unsqueeze(-1) * rgb, dim=1)  # (N_rays, 3)
        depth_values = torch.sum(weights * depth, dim=1)          # (N_rays,)

        # 白色背景处理
        if self.white_bkgd:
            rgb_values = rgb_values + (1.0 - weights.sum(dim=-1, keepdim=True))

        return rgb_values, depth_values

        


    