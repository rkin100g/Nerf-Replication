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
        """Prepares inputs and applies network 'fn'."""
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