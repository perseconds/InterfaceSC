import argparse
import pandas as pd
import torch
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from Net.net import ChannelCode
from util import *
from data.dataset import ImageNet32
from torch.utils.data import ConcatDataset
parser = argparse.ArgumentParser()
parser.add_argument('--trainset', type=str, default='CIFAR10', choices=['CIFAR10', 'ImageNet32', 'SVHN', 'CIFAR100'])
parser.add_argument('--channel', type=str, default='awgn', choices=['awgn', 'rayleigh'])
parser.add_argument('--C', type=int, default=4)

args = parser.parse_args()

class config():
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    downsample = 2
    #设置参数
    epoch_num = 300
    lr = 1e-4
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
        epoch_num = 80
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
        file_path = checkpoints_path + "/eval_result_CIFAR10.csv"
    elif args.trainset == 'SVHN':
        epoch_num = 80
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
        file_path = checkpoints_path + "/eval_result_SVHN.csv"
    elif args.trainset == 'ImageNet32':
        epoch_num = 80
        lr = 5e-4
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
       
        checkpoints_path = "./1checkpoints_channel_ImageNet32_" + args.channel  + str(args.C)
        source_path = "./checkpoints_source_ImageNet32"
        file_path = checkpoints_path + "/eval_result_ImageNet32.csv"
    elif args.trainset == 'CIFAR100':
        # CIFAR-10 数据集的均值和标准差
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
            #加载数据
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)  # 归一化到 [-1, 1]
        ])
        train_dataset = datasets.CIFAR100("./data", train=True, transform=transform, download=True)
        test_dataset = datasets.CIFAR100("./data", train=False, transform=transform, download=True)
        # print(train_dataset[0][1].shape)
        batchsize = 128
        train_iter = DataLoader(train_dataset, batch_size=batchsize , shuffle=True)
        test_iter = DataLoader(test_dataset, batch_size=batchsize , shuffle=True)

        checkpoints_path = "./checkpoints_channel_CIFAR10_" + args.channel + str(args.C)
        source_path = "./checkpoints_source_CIFAR10"
        file_path = checkpoints_path + "/eval_result_CIFAR100.csv"
   
model = ChannelCode(config, args).to(config.device)
#加载信源编解码器
model_dict = model.state_dict()
pretrained_dict = torch.load(config.source_path + "/checkpoints_best.pth",  weights_only=False)
pretrained_dict = {
    k: v for k, v in pretrained_dict.items() 
    if k in model_dict and model_dict[k].shape == v.shape
    }
model_dict.update(pretrained_dict)
#加载信道编解码器
channel_dict = torch.load(config.checkpoints_path + "/checkpoints_best.pth",  weights_only=False)
channel_dict = {
    k: v for k, v in channel_dict.items() 
    if k in model_dict and model_dict[k].shape == v.shape
    }
model_dict.update(channel_dict)
model.load_state_dict(model_dict, strict=False)

snr_list = [5,8,10,13,15,18,20]
img_psnr_list = []
for snr in snr_list:
    img_psnr = 0
    test_length = 0
    model.eval()
    for img, label in config.test_iter:
        img = img.to(config.device)
        img_recontrust, CBR = model(img, snr = snr)
        # l = loss1(img_recontrust, img)
        # l = loss1(img_recontrust, img)

        psnr = calculate_psnr(img, img_recontrust)

        img_psnr += psnr.item() * img.shape[0]
        # test_loss += l.item() * img.shape[0]
        test_length += img.shape[0]
    print(f"snr:{snr}, psnr:{img_psnr/test_length:f}, CBR:{CBR}")
    img_psnr_list.append(img_psnr/test_length)


data = pd.DataFrame({'snr': snr_list, 'psnr': img_psnr_list, 'CBR': CBR})
data.to_csv(config.file_path, index=False)
