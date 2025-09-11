import torch
import torch.nn as nn
import numpy as np
from torch.nn import functional as F
from src.models.encoding import get_encoder
from src.config import cfg


class NeRF(nn.Module):
    def __init__(
        self, D=8, W=256, input_ch=3, input_ch_views=3, skips=[4], use_viewdirs=False
    ):
        super(NeRF, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.skips = skips
        self.use_viewdirs = use_viewdirs
        self.output_ch = 5 if self.use_viewdirs else 4

        self.pts_linears = nn.ModuleList(
            [nn.Linear(self.input_ch, self.W)]
            + [
                (
                    nn.Linear(self.W, self.W)
                    if i not in self.skips
                    else nn.Linear(self.W + self.input_ch, self.W)
                )
                for i in range(self.D - 1)
            ]
        )

        self.views_linears = nn.ModuleList(
            [nn.Linear(self.input_ch_views + self.W, self.W // 2)]
        )

        if self.use_viewdirs:
            # feature vector(256)
            self.feature_linear = nn.Linear(self.W, self.W)
            # alpha(1)
            self.alpha_linear = nn.Linear(self.W, 1)
            # rgb color(3)
            self.rgb_linear = nn.Linear(self.W // 2, 3)
        else:
            # output channel(default: 4)
            self.output_linear = nn.Linear(self.W, self.output_ch)

    def forward(self, x):
        input_pts, input_views = torch.split(
            x, [self.input_ch, self.input_ch_views], dim=-1
        )
        h = input_pts
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([input_pts, h], -1)

        if self.use_viewdirs:
            alpha = self.alpha_linear(h)
            feature = self.feature_linear(h)
            h = torch.cat([feature, input_views], -1)

            for i, l in enumerate(self.views_linears):
                h = self.views_linears[i](h)
                h = F.relu(h)

            rgb = self.rgb_linear(h)
            outputs = torch.cat([rgb, alpha], -1)
        else:
            outputs = self.output_linear(h)

        return outputs

    def load_weights_from_keras(self, weights):
        assert self.use_viewdirs, "Not implemented if use_viewdirs=False"

        # Load pts_linears
        for i in range(self.D):
            idx_pts_linears = 2 * i
            self.pts_linears[i].weight.data = torch.from_numpy(
                np.transpose(weights[idx_pts_linears])
            )
            self.pts_linears[i].bias.data = torch.from_numpy(
                np.transpose(weights[idx_pts_linears + 1])
            )

        # Load feature_linear
        idx_feature_linear = 2 * self.D
        self.feature_linear.weight.data = torch.from_numpy(
            np.transpose(weights[idx_feature_linear])
        )
        self.feature_linear.bias.data = torch.from_numpy(
            np.transpose(weights[idx_feature_linear + 1])
        )

        # Load views_linears
        idx_views_linears = 2 * self.D + 2
        self.views_linears[0].weight.data = torch.from_numpy(
            np.transpose(weights[idx_views_linears])
        )
        self.views_linears[0].bias.data = torch.from_numpy(
            np.transpose(weights[idx_views_linears + 1])
        )

        # Load rgb_linear
        idx_rbg_linear = 2 * self.D + 4
        self.rgb_linear.weight.data = torch.from_numpy(
            np.transpose(weights[idx_rbg_linear])
        )
        self.rgb_linear.bias.data = torch.from_numpy(
            np.transpose(weights[idx_rbg_linear + 1])
        )

        # Load alpha_linear
        idx_alpha_linear = 2 * self.D + 6
        self.alpha_linear.weight.data = torch.from_numpy(
            np.transpose(weights[idx_alpha_linear])
        )
        self.alpha_linear.bias.data = torch.from_numpy(
            np.transpose(weights[idx_alpha_linear + 1])
        )


class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.N_samples = cfg.task_arg.N_samples
        self.N_importance = cfg.task_arg.N_importance
        self.chunk = cfg.task_arg.chunk_size
        self.batch_size = cfg.task_arg.N_rays
        self.white_bkgd = cfg.task_arg.white_bkgd
        self.use_viewdirs = cfg.task_arg.use_viewdirs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.sample_size = getattr(cfg, "sample_size", 64)
        self.rays_size = getattr(cfg, "rays_size", 160000)

        # encoder
        self.embed_fn, self.input_ch = get_encoder(cfg.network.xyz_encoder)
        self.embeddirs_fn, self.input_ch_views = get_encoder(cfg.network.dir_encoder)

        # coarse model
        self.model = NeRF(
            D=cfg.network.nerf.D,
            W=cfg.network.nerf.W,
            input_ch=self.input_ch,
            input_ch_views=self.input_ch_views,
            skips=cfg.network.nerf.skips,
            use_viewdirs=self.use_viewdirs,
        )

        # fine model
        self.model_fine = NeRF(
            D=cfg.network.nerf.D,
            W=cfg.network.nerf.W,
            input_ch=self.input_ch,
            input_ch_views=self.input_ch_views,
            skips=cfg.network.nerf.skips,
            use_viewdirs=self.use_viewdirs,
        )

    def batchify(self, fn, chunk):
        """Constructs a version of 'fn' that applies to smaller batches."""

        def ret(inputs):
            return torch.cat(
                [fn(inputs[i : i + chunk]) for i in range(0, inputs.shape[0], chunk)], 0
            )

        return ret

    """
    def forward(self, inputs, viewdirs, model=""):
        Prepares inputs and applies network 'fn'.
        if model == "fine":
            fn = self.model_fine
        else:
            fn = self.model

        inputs_flat = torch.reshape(inputs, [-1, inputs.shape[-1]])
        embedded = self.embed_fn(inputs_flat)

        if self.use_viewdirs:
            input_dirs = viewdirs[:, None].expand(inputs.shape)
            input_dirs_flat = torch.reshape(input_dirs, [-1, input_dirs.shape[-1]])
            embedded_dirs = self.embeddirs_fn(input_dirs_flat)
            embedded = torch.cat([embedded, embedded_dirs], -1)

        embedded = embedded.to(torch.float32)
        outputs_flat = self.batchify(fn, self.chunk)(embedded)
        outputs = torch.reshape(
            outputs_flat, list(inputs.shape[:-1]) + [outputs_flat.shape[-1]]
        )
        return outputs
    """


    def forward(self, inputs, viewdirs, valid_mask, model=""):
        if model == "fine":
            fn = self.model_fine
        else:
            fn = self.model

        # Part 1：构造输入
        # 筛选有效点
        if valid_mask is not None:
            # 展平掩码
            valid_mask_flat = valid_mask.reshape(-1)
            # 筛选有效点
            inputs_valid = inputs.reshape(-1, 3)[valid_mask_flat]
            # 先拓展视角再筛选有效视角
            viewdirs_expand = viewdirs[:, None].expand_as(inputs)  # [N_rays, N_importance, 3]
            viewdirs_valid = viewdirs_expand.reshape(-1, 3)[valid_mask_flat]
        else:
            inputs_valid = torch.reshape(inputs, [-1, inputs.shape[-1]])
            viewdirs_valid = viewdirs[:, None].expand_as(inputs).reshape(-1, 3)

        # 对坐标编码
        embedded = self.embed_fn(inputs_valid)

        if self.use_viewdirs:
            # 对视角编码
            embedded_dirs = self.embeddirs_fn(viewdirs_valid)
            embedded = torch.cat([embedded, embedded_dirs], -1)

        embedded = embedded.to(torch.float32)
        
        # Part 2：分块处理计算结果
        outputs_valid_flat = self.batchify(fn, self.chunk)(embedded)

        # Part 3：构造输出
        # Step 1:初始化原形状的输出张量（无效点用0填充，不影响体渲染）
        outputs_shape = list(inputs.shape[:-1]) + [outputs_valid_flat.shape[-1]]  # [N_rays, N_importance, output_ch]
        outputs = torch.zeros(outputs_shape, device=self.device)

        # Step 2:
        if valid_mask is not None:
            # 获取shape
            N_rays, N_importance = valid_mask.shape
            
            # 生成全局展平索引：[0, 1, ..., N_rays*N_importance-1]
            global_indices = torch.arange(N_rays * N_importance, device=self.device)
            
            # 筛选有效点的全局索引
            valid_global_indices = global_indices[valid_mask_flat]
            
            # 转换为原张量的2D索引（r=射线号，s=该射线内的采样点号）
            r = valid_global_indices // N_importance
            s = valid_global_indices % N_importance
            
            # 3. 将有效点结果填充到原输出张量
            outputs[r, s] = outputs_valid_flat
        else:
            # 无无效点，直接重构维度
            outputs = torch.reshape(outputs_valid_flat, outputs_shape)

        return outputs

    """
    def forward(self, inputs, viewdirs, valid_mask, depth, model="",ert_threshold=0.01):
        if model == "fine":
            fn = self.model_fine
        else:
            fn = self.model

        # 获取shape
        N_rays = inputs.shape[0]
        N_importance = inputs.shape[1]

        # 视角拓展
        viewdirs_expand = viewdirs[:, None].expand_as(inputs)  # [N_rays, N_importance, 3]
        
        # 初始化射线级状态：活跃射线掩码+透射率
        active_rays = torch.ones(N_rays, dtype=torch.bool, device=self.device)  # 初始所有射线活跃
        T = torch.ones(N_rays, 1, device=self.device)  # 初始透射率（每条射线1个值）

        # 初始化输出形状
        output_ch = 4
        outputs = torch.zeros(N_rays, N_importance, output_ch, device=self.device)
        
        # 生成全局展平索引：[0, 1, ..., N_rays*N_importance-1]用于后续保存
        global_indices = torch.arange(N_rays * N_importance, device=self.device)
        
        # 分块处理

        if valid_mask is not None:
            for ray_start in range(0,inputs.shape[0],self.rays_size):
                # 获取当前处理射线块索引
                ray_end = min(ray_start + self.rays_size, N_rays)
                current_ray_slice = slice(ray_start, ray_end)  # 当前处理的射线范围

                # 过滤当前射线块中的活跃射线（仅处理未终止的射线）
                current_active = active_rays[current_ray_slice]  # [current_ray_count]
                if not current_active.any():
                    continue  # 该射线块全终止，直接跳过
                
                # 提取当前射线块中的活跃射线索引（相对射线块的局部索引）
                active_idx_in_block = torch.where(current_active)[0]  # [active_in_block_count]
                total_active = active_idx_in_block.numel()
                if total_active == 0:
                    continue
                
                for sample_start in range(0, inputs.shape[1], self.sample_size):
                    # Part 1：确定采样点范围
                    sample_end = min(sample_start + self.sample_size, N_importance)
                    current_sample_slice = slice(sample_start, sample_end)  # 当前处理的采样点范围
                    sample_count = sample_end - sample_start  # 本块采样点数量
                    
                    # Part 2：获取活跃射线的采样点的掩码
                    current_valid_mask = valid_mask[current_ray_slice, current_sample_slice]  # [current_ray_count, sample_count]
                    current_mask = current_valid_mask[active_idx_in_block]  # [active_in_block_count, sample_count]
                    if not current_mask.any():
                        continue  # 本块无有效点，跳过
                    
                    # Part 3：获取坐标和视角（使用全局索引）
                    # Step 1：获得有效点的ray和sample局部索引
                    ray_idx_in_block, sample_idx_in_block = torch.where(current_mask)  # 均为[valid_count]
                    
                    # Step 2：转换为全局射线号（原始射线索引）
                    global_ray_idx = ray_start + active_idx_in_block[ray_idx_in_block]  # [valid_count]
                    
                    # Step 3：转换为全局采样点号（在射线内的位置）
                    global_sample_idx = sample_start + sample_idx_in_block  # [valid_count]

                    # Step 4：提取坐标和视角
                    inputs_valid = inputs[global_ray_idx, global_sample_idx].reshape(-1, 3)  # [valid_count, 3]
                    viewdirs_valid = viewdirs_expand[global_ray_idx, global_sample_idx].reshape(-1, 3)  # [valid_count, 3]
                    
                    # Part 4:编码
                    chunk_size = self.chunk  # 建议设为1024/2048（根据显存调整，越小越安全）
                    outputs_valid_list = []  # 存储每块的结果

                    # 逐块处理有效点：编码→网络计算
                    for i in range(0, inputs_valid.shape[0], chunk_size):
                        # 1. 提取当前小分块的输入和视角
                        chunk_input = inputs_valid[i:i+chunk_size]  # [chunk_size, 3]
                        chunk_viewdir = viewdirs_valid[i:i+chunk_size]  # [chunk_size, 3]
                        
                        # 2. 逐块编码（小张量，显存压力小）
                        chunk_embedded = self.embed_fn(chunk_input)
                        if self.use_viewdirs:
                            chunk_embedded_dir = self.embeddirs_fn(chunk_viewdir)
                            chunk_embedded = torch.cat([chunk_embedded, chunk_embedded_dir], -1)
                        chunk_embedded = chunk_embedded.to(torch.float32)
                        
                        # 3. 逐块网络计算（可复用原batchify，也可直接计算）
                        # 若chunk_size已很小（如1024），可直接调用fn，无需再batchify
                        chunk_output = fn(chunk_embedded)
                        # 若仍OOM，可继续用batchify（双重保险）：
                        # chunk_output = self.batchify(fn, self.chunk//4)(chunk_embedded)
                        
                        outputs_valid_list.append(chunk_output)

                    # 合并所有小分块的结果（恢复原形状）
                    outputs_valid = torch.cat(outputs_valid_list, dim=0)  # [valid_count, 4]

                    # Part 5:网络处理
                    #outputs_valid = self.batchify(fn, self.chunk)(embedded)

                    # Part 4：保存结果
                    outputs[global_ray_idx, global_sample_idx] = outputs_valid
                    
                    # 透射率更新与光线筛选（核心修正部分）
                    # 1. 提取密度并按射线分组
                    density_flat = F.relu(outputs[global_ray_idx, global_sample_idx, 3])  # [valid_count]
                    ray_counts = torch.bincount(ray_idx_in_block, minlength=total_active)  # 每条射线的有效点数量
                    density_split = density_flat.split(ray_counts.tolist())  # 按射线拆分
                    
                    # 2. 提取深度并按射线分组
                    depth_flat = depth[global_ray_idx, global_sample_idx]  # [valid_count]
                    depth_split = depth_flat.split(ray_counts.tolist())  # 按射线拆分
                    
                    # 3. 初始化每条射线的透射率（当前块的起始T）
                    # 从全局T中提取每条活跃射线的初始透射率（用ray_starts定位）
                    ray_starts = torch.cat([torch.tensor([0], device=self.device), torch.cumsum(ray_counts, dim=0)[:-1]])
                    unique_global_ray_idx = global_ray_idx[ray_starts]  # 每条射线的全局索引（去重）
                    T_per_ray = T[unique_global_ray_idx].clone()  # [active_in_block_count, 1]
                    
                    # 4. 逐射线计算累积透射率并判断终止
                    terminated = torch.zeros(total_active, dtype=torch.bool, device=self.device)  # 标记当前块中终止的射线
                    
                    for ray_i in range(total_active):
                        # 4.1 当前射线的有效点数据
                        d = density_split[ray_i]
                        z = depth_split[ray_i]
                        num_valid = d.numel()
                        if num_valid == 0:
                            continue  # 无有效点，跳过
                        
                        # 4.2 计算delta
                        delta = z[1:] - z[:-1] if num_valid > 1 else torch.tensor([], device=self.device)
                        delta = torch.cat([delta, 1e10 * torch.ones(1, device=self.device)])  # 补最后一个点
                        
                        # 4.3 计算累积透射率
                        alpha = 1.0 - torch.exp(-d * delta)
                        cumprod_alpha = torch.cumprod(1 - alpha, dim=0)  # [num_valid]
                        T_batch = T_per_ray[ray_i] * cumprod_alpha  # 该射线在当前块的累积透射率
                        
                        # 4.4 判断是否终止（找到第一个<阈值的点）
                        terminate_mask = T_batch < ert_threshold
                        if terminate_mask.any():
                            # 取第一个终止点的透射率作为最终值
                            first_terminate_idx = torch.argmax(terminate_mask.float())
                            T_per_ray[ray_i] = T_batch[first_terminate_idx]
                            terminated[ray_i] = True  # 标记该射线终止
                        else:
                            # 未终止，取最后一个点的透射率
                            T_per_ray[ray_i] = T_batch[-1]
                    
                    # 5. 更新全局透射率和活跃射线掩码
                    # 5.1 更新透射率（仅活跃射线）
                    T[unique_global_ray_idx] = T_per_ray
                    
                    # 5.2 标记终止射线（转换为全局索引）
                    if terminated.any():
                        terminated_global_idx = unique_global_ray_idx[terminated]
                        active_rays[terminated_global_idx] = False

        # 仅保留Forward内部的一次分块（射线块），采样点不再额外分块（或合并到射线块内）
        if valid_mask is not None:
            for ray_start in range(0, N_rays, self.rays_size):
                ray_end = min(ray_start + self.rays_size, N_rays)
                current_ray_slice = slice(ray_start, ray_end)
                current_active = active_rays[current_ray_slice]
                if not current_active.any():
                    continue

                # 1. 提取当前块的活跃射线数据（射线级：[active_count, N_importance, 3]）
                active_idx_in_block = torch.where(current_active)[0]
                total_active = active_idx_in_block.numel()
                global_ray_idx = ray_start + active_idx_in_block  # 活跃射线的全局索引

                # 2. 提取当前块的所有有效采样点（展平为[total_valid, ...]，避免逐点索引）
                # 2.1 有效点掩码：[active_count, N_importance] → 展平为[total_valid]
                current_valid_mask = valid_mask[global_ray_idx]  # [active_count, N_importance]
                valid_flat_mask = current_valid_mask.view(-1)  # [active_count * N_importance]
                if not valid_flat_mask.any():
                    continue

                # 2.2 提取有效点的坐标、视角、深度（展平为大张量，减少小张量操作）
                inputs_block = inputs[global_ray_idx].view(-1, 3)  # [active_count*N_importance, 3]
                viewdirs_block = viewdirs_expand[global_ray_idx].view(-1, 3)  # 同上
                depth_block = depth[global_ray_idx].view(-1)  # [active_count*N_importance]

                inputs_valid = inputs_block[valid_flat_mask]  # [total_valid, 3]
                viewdirs_valid = viewdirs_block[valid_flat_mask]  # [total_valid, 3]
                depth_valid = depth_block[valid_flat_mask]  # [total_valid]

                # 3. 批量编码+网络推理（大张量操作，效率高）
                embedded = self.embed_fn(inputs_valid)
                if self.use_viewdirs:
                    embedded_dirs = self.embeddirs_fn(viewdirs_valid)
                    embedded = torch.cat([embedded, embedded_dirs], -1)
                embedded = embedded.to(torch.float32)

                # 批量推理（chunk_size设为2048，平衡显存和速度）
                outputs_valid = []
                for i in range(0, embedded.shape[0], self.chunk):
                    chunk_out = fn(embedded[i:i+self.chunk])
                    outputs_valid.append(chunk_out)
                outputs_valid = torch.cat(outputs_valid, dim=0)  # [total_valid, 4]

                # 4. 向量式ERT计算（核心优化：无Python循环）
                # 4.1 准备基础数据：密度、射线ID、点ID
                density_valid = F.relu(outputs_valid[:, 3])  # [total_valid]
                # 每个有效点对应的「射线在当前块的索引」（0~total_active-1）
                ray_ids = torch.repeat_interleave(
                    torch.arange(total_active, device=self.device),
                    current_valid_mask.sum(dim=1)  # 每条射线的有效点数量
                )
                # 每个有效点对应的「在射线内的位置索引」（0~N_importance-1）
                point_ids = torch.cat([
                    torch.where(mask)[0] for mask in current_valid_mask
                ], dim=0)  # [total_valid]

                # 4.2 批量计算delta（相邻采样点深度差，最后一个点补1e10）
                # 计算每个有效点的下一个点深度（无下一个则补1e10）
                next_depth = torch.roll(depth_valid, shifts=-1, dims=0)
                # 标记每条射线的最后一个有效点（这些点的next_depth设为1e10）
                last_point_mask = (point_ids == current_valid_mask.sum(dim=1)[ray_ids] - 1)
                next_depth[last_point_mask] = 1e10
                delta = next_depth - depth_valid  # [total_valid]

                # 4.3 批量计算alpha和累积透射率
                alpha = 1.0 - torch.exp(-density_valid * delta)  # [total_valid]
                # 批量计算每条射线的累积透射率（1 - alpha的累积积）
                # 技巧：用pad隔离不同射线，避免跨射线累积
                pad_values = torch.ones(total_active, device=self.device)  # 每条射线前加1（累积初始值）
                # 按射线分组，每个组前插入1，再计算cumprod
                ray_valid_counts = current_valid_mask.sum(dim=1)  # [total_active]
                blocks = []
                start = 0
                for cnt in ray_valid_counts:
                    if cnt == 0:
                        blocks.append(torch.tensor([1.0], device=self.device))
                        continue
                    # 当前射线的(1 - alpha)
                    block_alpha = 1.0 - alpha[start:start+cnt]
                    # 插入初始值1，再cumprod
                    block = torch.cat([torch.tensor([1.0], device=self.device), block_alpha])
                    blocks.append(block)
                    start += cnt
                # 合并所有块并计算cumprod，再去掉初始值1
                pad_alpha = torch.cat(blocks, dim=0)
                cumprod_alpha = torch.cumprod(pad_alpha, dim=0)[1:]  # [total_valid]

                # 4.4 批量判断射线终止并更新透射率
                # 初始透射率（每条射线的当前T值）
                T_per_ray = T[global_ray_idx].squeeze(1)  # [total_active]
                # 每条有效点的透射率：T_per_ray[ray_ids] * cumprod_alpha
                T_batch = T_per_ray[ray_ids] * cumprod_alpha  # [total_valid]

                # 批量找每条射线的第一个终止点（T_batch < 阈值）
                # 权重：终止点设为「倒序索引」（第一个终止点权重最大），非终止点设0
                weight = torch.where(
                    T_batch < ert_threshold,
                    torch.arange(len(T_batch), 0, -1, device=self.device).float(),
                    0.0
                )
                # 按射线分组，取权重最大的点（即第一个终止点）
                ray_terminate_idx = torch.full((total_active,), -1, device=self.device)
                ray_terminated = torch.zeros(total_active, dtype=torch.bool, device=self.device)
                
                start = 0
                for ray_i in range(total_active):
                    cnt = ray_valid_counts[ray_i]
                    if cnt == 0:
                        start += cnt
                        continue
                    ray_weight = weight[start:start+cnt]
                    if ray_weight.max() > 0:
                        # 有终止点，取最大权重对应的索引
                        idx_in_ray = torch.argmax(ray_weight)
                        ray_terminate_idx[ray_i] = start + idx_in_ray
                        ray_terminated[ray_i] = True
                    start += cnt

                # 4.5 更新透射率和活跃射线掩码
                # 选择更新值：终止则用第一个终止点的T，否则用最后一个点的T
                last_point_idx = torch.cumsum(ray_valid_counts, 0) - 1  # 每条射线最后一个有效点索引
                select_idx = torch.where(ray_terminated, ray_terminate_idx, last_point_idx)
                T_per_ray_updated = T_batch[select_idx]  # [total_active]

                # 更新全局透射率和活跃射线
                T[global_ray_idx] = T_per_ray_updated.unsqueeze(1)
                active_rays[global_ray_idx[ray_terminated]] = False

                # 5. 保存输出结果（将valid结果写回全局outputs）
                # 构建全局索引：(global_ray_idx[ray_ids], point_ids)
                outputs[global_ray_idx[ray_ids], point_ids] = outputs_valid
        
        else:
            inputs_flat = torch.reshape(inputs, [-1, inputs.shape[-1]])
            embedded = self.embed_fn(inputs_flat)

            if self.use_viewdirs:
                input_dirs = viewdirs[:, None].expand(inputs.shape)
                input_dirs_flat = torch.reshape(input_dirs, [-1, input_dirs.shape[-1]])
                embedded_dirs = self.embeddirs_fn(input_dirs_flat)
                embedded = torch.cat([embedded, embedded_dirs], -1)

            embedded = embedded.to(torch.float32)
            outputs_flat = self.batchify(fn, self.chunk)(embedded)
            outputs = torch.reshape(
                outputs_flat, list(inputs.shape[:-1]) + [outputs_flat.shape[-1]]
            )

        return outputs
        """
