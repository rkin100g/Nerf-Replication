import torch
import torch.nn as nn
from src.models.nerf.renderer.volume_renderer import Renderer
import torch.nn.functional as F
import torchmetrics



class NetworkWrapper(nn.Module):
    def __init__(self, net, train_loader):
        super(NetworkWrapper, self).__init__()
        self.net = net
        self.renderer = Renderer(self.net)

        # add metrics here
         # Loss function (NeRF uses L2 loss)
        self.criterion = nn.MSELoss()

        # Metrics
        self.psnr = torchmetrics.PeakSignalNoiseRatio(data_range=1.0)

    def forward(self, batch):
        """
        Write your codes here.
        """
        # 预测值
        rgb_pred, depth_pred = self.renderer.render(batch)

        # 真实值
        rgb_gt = batch["colors"]  

        # loss计算
        loss = self.criterion(rgb_pred, rgb_gt)

        # 计算 Metrics
        psnr_val = self.psnr(rgb_pred, rgb_gt)
        
        # 返回结果字典
        return {
            "loss": loss,
            "rgb_pred": rgb_pred,
            "rgb_gt": rgb_gt,
            "depth_pred": depth_pred,
            "psnr": psnr_val,
        }

