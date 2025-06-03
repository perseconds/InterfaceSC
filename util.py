'''
Author: kwz
Date: 2025-03-14 20:32:25
LastEditors: kwz
LastEditTime: 2025-06-03 14:56:24
Description: 计算PSNR和MSEloss
'''
import os
import random
import numpy as np
import torch.nn.functional as F
import torch
def calculate_psnr(img1, img2, normalization=True):
    """
    计算图像的 PSNR
    """
    # 计算均方误差 (MSE)
    #[-1, 1] -> [0, 1]
    if normalization:
        img1 = (img1 + 1) / 2
        img2 = (img2 + 1) / 2
    mse = F.mse_loss(img1 * 255., img2 * 255.)
    
    # 计算 PSNR
    psnr = 10 * torch.log10(255.**2 / mse)
    
    return psnr
class MSE(torch.nn.Module):
    def __init__(self, normalization=True):
        super(MSE, self).__init__()
        self.squared_difference = torch.nn.MSELoss(reduction='mean')
        self.normalization = normalization

    def forward(self, X, Y):
        # [-1 1] to [0 1]
        if self.normalization:
            X = (X + 1) / 2
            Y = (Y + 1) / 2
        return self.squared_difference(X * 255., Y * 255.)  # / 255.


class DynamicRandomSampler(torch.utils.data.Sampler):
    def __init__(self, dataset_size, num_samples_per_epoch, seed=None):
        self.dataset_size = dataset_size
        self.num_samples = num_samples_per_epoch
        self.seed = seed
        self.generator = torch.Generator()
        if seed is not None:
            self.generator.manual_seed(seed)

    def __iter__(self):
        # 每个epoch生成新的随机索引
        indices = torch.randperm(self.dataset_size, generator=self.generator)[:self.num_samples]
        return iter(indices.tolist())

    def __len__(self):
        return self.num_samples
def seed_torch(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)  # 为了禁止hash随机化，使得实验可复现
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
