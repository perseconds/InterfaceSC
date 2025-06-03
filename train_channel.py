import argparse
import os
import random
import pandas as pd
import torch
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from Net.net import ChannelCode
from tqdm import tqdm
from util import *
from torch.optim.lr_scheduler import MultiStepLR
from data.dataset import ImageNet32,TinyImageNet
from torch.utils.data import ConcatDataset

parser = argparse.ArgumentParser()
parser.add_argument('--trainset', type=str, default='CIFAR10', choices=['CIFAR10', 'SVHN', 'ImageNet32'])
parser.add_argument('--channel', type=str, default='awgn', choices=['awgn', 'rayleigh'])
parser.add_argument('--C', type=int, default=4)


args = parser.parse_args()
seed_torch()
class config():

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #设置参数
    downsample = 2
    epoch_num = 80
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
    chan_en = dict(
        img_size = (8, 8), input_dims = 96,
                embed_dims = 128, depths=2, num_heads = 4, C = args.C*8*8,
                mlp_ratio=4., qkv_bias=True,
                norm_layer=nn.LayerNorm
    )
    chan_de = dict(
        img_size = (8, 8), input_dims = args.C*8*8,
                embed_dims = 128, depths=2, num_heads = 4, C = 96,
                mlp_ratio=4., qkv_bias=True,
                norm_layer=nn.LayerNorm
                )
    if args.trainset == 'CIFAR10':
        # epoch_num = 80
        lr = 1e-4
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
       

        checkpoints_path = "./checkpoints_channel_CIFAR10_" + args.channel + str(args.C)
        source_path = "./checkpoints_source_CIFAR10"
    elif args.trainset == 'SVHN':
        # epoch_num = 80
        lr = 1e-4
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
    
        checkpoints_path = "./checkpoints_channel_SVHN_" + args.channel  + str(args.C)
        source_path = "./checkpoints_source_SVHN"
    elif args.trainset == 'ImageNet32':
        # epoch_num = 80
        lr = 1e-4
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
       
        checkpoints_path = "./checkpoints_channel_ImageNet32_" + args.channel  + str(args.C)
        source_path = "./checkpoints_source_ImageNet32"



#设置参数
loss1 = MSE(normalization=True)
model = ChannelCode(config, args).to(config.device)
#加载编解码器
model_dict = model.state_dict()
pretrained_dict = torch.load(config.source_path + "/checkpoints_best.pth",  weights_only=False)
pretrained_dict = {
    k: v for k, v in pretrained_dict.items() 
    if k in model_dict and model_dict[k].shape == v.shape
    }
model_dict.update(pretrained_dict)
model.load_state_dict(model_dict, strict=False)
optimizer1 = torch.optim.Adam(list(model.chan_encoder.parameters()) + list(model.chan_decoder.parameters()), lr=config.lr,  weight_decay=1e-5)
# 开始训练
train_loss_list = []
test_loss_list = []
snr_list = [5,8,10,13,15,18,20]
record_psnr = 0
for epoch in range(config.epoch_num):
    train_loss = 0
    train_length = 0

    model.train()
    for img, label in tqdm(config.train_iter):
        img = img.to(config.device)

        snr = random.choice(snr_list)
        # snr = 20
        img_recontrust, _ = model(img, snr)
        l = loss1(img_recontrust, img)

        optimizer1.zero_grad()
        l.backward()
        optimizer1.step()

        train_loss += l.item() * img.shape[0]
        train_length += img.shape[0]

    model.eval()
    test_loss = 0
    test_length = 0
    img_psnr = 0
    for img, label in tqdm(config.test_iter):
        img = img.to(config.device)
        img_recontrust, CBR = model(img, 20)
        # l = loss1(img_recontrust, img)
        l = loss1(img_recontrust, img)

        psnr = calculate_psnr(img, img_recontrust)

        img_psnr += psnr.item() * img.shape[0]
        test_loss += l.item() * img.shape[0]
        test_length += img.shape[0]
        # img_psnr_list.append(img_psnr/test_length)
    psnr_avg = img_psnr/test_length
    # scheduler.step()
    if psnr_avg > record_psnr:
        record_psnr = psnr_avg
        if not os.path.exists(config.checkpoints_path):
            os.makedirs(config.checkpoints_path)
        with open(config.checkpoints_path + f'/checkpoints_best.pth', 'wb') as f:
            torch.save(model.state_dict(), f)
        with open(config.checkpoints_path+"/result_psnr.txt", "w", encoding="utf-8") as f:
            f.write(str(record_psnr))  # 将变量转换为字符串后写入
    print(f"epoch:{epoch + 1}, train_loss: {train_loss / train_length:f},  test_loss:{test_loss / test_length:f}")
    print(f"PSNR:{psnr_avg}, CBR:{CBR}")#
