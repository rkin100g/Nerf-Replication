import numpy as np
from src.config import cfg
import os
import torch.nn.functional as F
from skimage.metrics import structural_similarity as compare_ssim
import cv2
import json
import warnings
import torch

warnings.filterwarnings("ignore", category=UserWarning)


class Evaluator:
    def __init__(
        self,
    ):
        self.mse = []
        self.psnr = []
        self.ssim = []
        self.imgs = []

    def psnr_metric(self, img_pred, img_gt):
        img_pred = (np.clip(img_pred, 0, 1) * 255).astype(np.uint8)
        img_gt = (np.clip(img_gt, 0, 1) * 255).astype(np.uint8)
        mse = np.mean((img_pred - img_gt) ** 2)
        # 替换原有公式
        if mse < 1e-10:
            return 100.0
        return 10 * np.log10((255** 2) / mse)

    def ssim_metric_2(self, img_pred, img_gt, batch, id, num_imgs):
        result_dir = os.path.join(cfg.result_dir, "images")
        os.system("mkdir -p {}".format(result_dir))
        cv2.imwrite(
            "{}/view{:03d}_pred.png".format(result_dir, id),
            (img_pred[..., [2, 1, 0]] * 255),
        )
        cv2.imwrite(
            "{}/view{:03d}_gt.png".format(result_dir, id),
            (img_gt[..., [2, 1, 0]] * 255),
        )
        img_pred = (img_pred * 255).astype(np.uint8)

        ssim = compare_ssim(img_pred, img_gt, win_size=101, full=True)
        return ssim
    
    def ssim_metric(self, img_pred, img_gt, batch, id, num_imgs):
        result_dir = os.path.join(cfg.result_dir, "images")
        os.system("mkdir -p {}".format(result_dir))
        
        # 1. 保存图像（此处无问题，但可统一类型避免OpenCV警告）
        cv2.imwrite(
            "{}/view{:03d}_pred.png".format(result_dir, id),
            (img_pred[..., [2, 1, 0]] * 255).astype(np.uint8),  # 显式转uint8，消除警告
        )
        cv2.imwrite(
            "{}/view{:03d}_gt.png".format(result_dir, id),
            (img_gt[..., [2, 1, 0]] * 255).astype(np.uint8),    # 同样转uint8，统一保存格式
        )
        
        # 2. 统一img_pred和img_gt的类型与像素范围（关键修改1）
        img_pred = (img_pred * 255).astype(np.uint8)  # 原代码已有，保留
        img_gt = (img_gt * 255).astype(np.uint8)      # 新增：img_gt同步处理，确保与pred一致
        
        # 3. 修正SSIM计算参数（关键修改2：win_size缩小+指定channel_axis）
        # win_size设为7（默认值，兼容小分辨率），channel_axis=2（RGB通道在最后一维）
        ssim, ssim_map = compare_ssim(
            img_pred, 
            img_gt, 
            win_size=7,          # 缩小窗口尺寸，避免过大问题
            full=True,           # 原代码保留，返回完整SSIM图
            channel_axis=2       # 新增：指定多通道的通道维度，避免解析错误
        )
        
        return ssim  # 返回SSIM数值（原代码返回的是tuple，此处确保只返回数值，避免后续平均错误）

    def evaluate(self, output, batch):
        """
        Input:
            output: render输出的结果 rgb_pred,depth_pred
            batch:包括rays_o,rays_d,colors,H,W
        """
        # 数据获取与预处理(Pytorch张量转为np数组)
        rgb_pred, depth_pred = output      # (N_rays，3）
        rgb_gt = batch["colors"][..., :3]  # (B, N_rays, 3)
        rgb_gt = rgb_gt.reshape(-1,3)
        
        if torch.is_tensor(rgb_pred):
            rgb_pred = rgb_pred.detach().cpu().numpy()
        if torch.is_tensor(rgb_gt):
            rgb_gt = rgb_gt.detach().cpu().numpy()

        # 计算MSE和PSNR
        # 转为numpy后新增
        rgb_pred = np.clip(rgb_pred, 0, 1)
        rgb_gt = np.clip(rgb_gt, 0, 1)
        # 原有MSE计算保留
        mse = np.mean((rgb_pred - rgb_gt)** 2)
        print("mse:",mse)
        self.mse.append(mse)
        psnr_value = self.psnr_metric(rgb_pred, rgb_gt)
        print("psnr:",psnr_value)
        self.psnr.append(psnr_value)

        # 将(N_rays, 3)reshape为(H, W, 3)
        H = batch["H"]
        W = batch["W"]
        id = batch["id"]
        N_rays = rgb_pred.shape[0]
        assert H * W == N_rays, f"RGB维度不匹配:H*W={H*W} vs N_rays={N_rays}"
        
        # reshpae
        rgb_pred_reshaped = rgb_pred.reshape(H.item(), W.item(), 3)
        rgb_gt_reshaped = rgb_gt.reshape(H.item(), W.item(), 3)
        
        # 计算SSIM
        self.ssim.append(self.ssim_metric(rgb_pred_reshaped, rgb_gt_reshaped, batch, id.item(), 1))


    def summarize(self):
        mse_avg = np.mean(self.mse) if self.mse else 0.0
        psnr_avg = np.mean(self.psnr) if self.psnr else 0.0
        ssim_avg = np.mean(self.ssim) if self.ssim else 0.0
        
        return {
            "mse": float(mse_avg),
            "psnr": float(psnr_avg),
            "ssim": float(ssim_avg)
        }
