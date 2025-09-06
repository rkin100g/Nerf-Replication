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
        mse = np.mean((img_pred - img_gt) ** 2)
        psnr = -10 * np.log(mse) / np.log(10)
        return psnr

    def ssim_metric(self, img_pred, img_gt, batch, id, num_imgs):
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

    def evaluate(self, output, batch):
        
        rgb_pred, depth_pred = output
        rgb_gt = batch["colors"][..., :3]
        
        # 先转 numpy
        if torch.is_tensor(rgb_pred):
            rgb_pred = rgb_pred.detach().cpu().numpy()
        if torch.is_tensor(rgb_gt):
            rgb_gt = rgb_gt.detach().cpu().numpy()

        # 计算指标
        mse = np.mean((rgb_pred - rgb_gt) ** 2)

        self.mse.append(mse)

        psnr_val = self.psnr_metric(rgb_pred, rgb_gt)
        self.psnr.append(psnr_val)

        #ssim_val = self.ssim_metric(rgb_pred, rgb_gt)
        #self.ssim.append(ssim_val)

        # 保存图片
        result_dir = os.path.join(cfg.result_dir, "images")
        B, N_rays, _ = batch["rays_o"].shape
        H = W = 640
        rgb_pred = rgb_pred.reshape(H, W, 3)  # (H, W, 3)
        rgb_gt = rgb_gt.reshape(H, W, 3)  # (H, W, 3)

        os.makedirs(result_dir, exist_ok=True)
        cv2.imwrite(
            os.path.join(result_dir, f"view_pred.png"),
            (np.clip(rgb_pred,0,1)[..., ::-1] * 255).astype(np.uint8),
        )
        cv2.imwrite(
            os.path.join(result_dir, f"view_gt.png"),
            (np.clip(rgb_gt,0,1)[..., ::-1] * 255).astype(np.uint8),
        )

    def summarize(self):
        mse_avg = np.mean(self.mse) if self.mse else 0
        psnr_avg = np.mean(self.psnr) if self.psnr else 0
        #ssim_avg = np.mean(self.ssim) if self.ssim else 0
        return {
            "mse": mse_avg,
            "psnr": psnr_avg,
            #"ssim": ssim_avg
        }
