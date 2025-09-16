'''
Author: kwz
Date: 2025-03-26 22:10:29
LastEditors: kwz
LastEditTime: 2025-06-03 14:47:42
Description: bit重要性产生模块
'''
# date 2025/2/24
# author:kwz
# description: 依据变分方法推导的概率模型

import torch 
from torch import nn

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class BSC(nn.Module):
    def __init__(self, data_size):
        super().__init__()
        self.mu = nn.Parameter(torch.full(data_size, -4.))#训练开始时，使模型的误码率为0，先让编解码器学到有意义的表达
        self.sigmoid = nn.Sigmoid()
        
    def forward(self):
        noise_prob = self.sigmoid(self.mu)/2
        return noise_prob
