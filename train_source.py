import argparse
import os
import numpy as np
import pandas as pd
import torch
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader,SubsetRandomSampler
from Net.net import net
from tqdm import tqdm
from util import MSE, calculate_psnr, DynamicRandomSampler
from data.dataset import ImageNet32
from torch.utils.data import ConcatDataset

parser = argparse.ArgumentParser()
parser.add_argument('--trainset', type=str, default='CIFAR10', choices=['CIFAR10', 'ImageNet32', 'SVHN'])


args = parser.parse_args()

class config():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #设置参数
    # epoch_num = 500
    lr = 1e-4
    downsample = 2
    encoder_kwargs = dict(
        img_size=(32, 32), patch_size=2, in_chans=3,
        embed_dims=[128, 256], depths=[2, 4], num_heads=[4, 8], C=96,
        window_size=2, mlp_ratio=4., qkv_bias=True, qk_scale=None,
        norm_layer=nn.LayerNorm, patch_norm=True,
    )
    decoder_kwargs = dict(
        img_size=(32, 32),
        embed_dims=[256, 128], depths=[4, 2], num_heads=[8, 4], C=96,
        window_size=2, mlp_ratio=4., qkv_bias=True, qk_scale=None,
        norm_layer=nn.LayerNorm, patch_norm=True,
    ) 
    if args.trainset == 'CIFAR10':
        epoch_num = 250
    # CIFAR-10 数据集的均值和标准差
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
            #加载数据
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)  # 归一化到 [-1, 1]
        ])
        train_dataset = datasets.CIFAR10("./data", train=True, transform=transform, download=True)
        test_dataset = datasets.CIFAR10("./data", train=False, transform=transform, download=True)
        # print(train_dataset[0][1].shape)
        batchsize = 128
        train_iter = DataLoader(train_dataset, batch_size=batchsize , shuffle=True, num_workers=4)
        test_iter = DataLoader(test_dataset, batch_size=batchsize , shuffle=True, num_workers=4)
        

        checkpoints_path = "./checkpoints_source_CIFAR10"
    elif args.trainset == 'SVHN':
        epoch_num = 100
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
            #加载数据
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)  # 归一化到 [-1, 1]
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)  # 归一化到 [-1, 1]
        ])
        train_dataset = datasets.SVHN("./data", split='train',  transform=transform_train, download=True)
        test_dataset = datasets.SVHN("./data", split='test',  transform=transform_test, download=True)
        # print(train_dataset[0][1].shape)
        batchsize = 128
        train_iter = DataLoader(train_dataset, batch_size=batchsize , shuffle=True, num_workers=4)
        test_iter = DataLoader(test_dataset, batch_size=batchsize , shuffle=True, num_workers=4)

        checkpoints_path = "./checkpoints_source_SVHN"
    elif args.trainset == 'ImageNet32':
        epoch_num = 100
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
            #加载数据
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)  # 归一化到 [-1, 1]
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)  # 归一化到 [-1, 1]
        ])
        train_data_dir = './data'
        test_data_dir = './data'
        dataset_train = ImageNet32(train_data_dir, train=True, transform=transform_train)
        dataset_val = ImageNet32(test_data_dir, train=False, transform=transform_test)
        dataset_size = len(dataset_train)
        # 创建动态采样器（每个epoch选60000张不同的图）
        sampler = DynamicRandomSampler(dataset_size=dataset_size, num_samples_per_epoch=60000,seed=42)
        # print(train_dataset[0][1].shape)
        batchsize = 128
        train_iter = DataLoader(dataset_train, batch_size=batchsize , sampler=sampler, num_workers=4)
        test_iter = DataLoader(dataset_val, batch_size=batchsize , shuffle=True, num_workers=4)

        checkpoints_path = "./checkpoints_source_ImageNet32"





loss1 = MSE(normalization=True)
# loss2 = nn.CrossEntropyLoss()
loss2 = nn.MSELoss()
model = net(config).to(config.device)
optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
#-----------------存储路径设置-------------------
#-------------------------------------------------
# 开始训练
train_loss_list = []
test_loss_list = []
is_train = True
record_psnr = 0
for epoch in range(config.epoch_num):
    train_loss = 0
    train_length = 0
    test_loss = 0
    test_length = 0
    img_psnr = 0
    model.train()
    for img, label in tqdm(config.train_iter):
        img = img.to(config.device)
        img_recontrust, _, noise_prob = model(img)
        # print(torch.min(img))
        # print(torch.min(img_recontrust))
        l1 = loss1(img_recontrust, img)
        target = torch.full_like(noise_prob, 0.5)
        l2 = loss2(target, noise_prob)
        l = l1 + l2
        optimizer.zero_grad()
        l.backward()
        optimizer.step()
        train_loss += l.item() * img.shape[0]
        train_length += img.shape[0]
    model.eval()
    for img, label in tqdm(config.test_iter):
        img = img.to(config.device)
        img_recontrust, binary_code, noise_prob = model(img, is_train=False)
        l = loss1(img_recontrust, img)
        psnr = calculate_psnr(img, img_recontrust)

        img_psnr += psnr.item() * img.shape[0]
        test_loss += l.item() * img.shape[0]
        test_length += img.shape[0]
    if img_psnr/test_length > record_psnr:
        record_psnr = img_psnr/test_length
        if not os.path.exists(config.checkpoints_path):
            os.makedirs(config.checkpoints_path)
        with open(config.checkpoints_path + f'/checkpoints_best.pth', 'wb') as f:
            torch.save(model.state_dict(), f)
        with open(config.checkpoints_path+"/result_psnr.txt", "w", encoding="utf-8") as f:
            f.write(str(record_psnr))  # 将变量转换为字符串后写入
    # CBR = binary_code.numel()/(2 * img.numel())
    print(f"epoch:{epoch + 1}, train_loss: {train_loss / train_length:f}, test_loss:{test_loss / test_length:f}\
          PSNR:{img_psnr/test_length:f}, noise_max:{torch.max(noise_prob)}, noise_min:{torch.min(noise_prob)}")
    train_loss_list.append(train_loss / train_length)
    test_loss_list.append(test_loss / test_length)

