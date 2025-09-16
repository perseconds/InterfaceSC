'''
Author: kwz
Date: 2025-04-09 14:35:40
LastEditors: kwz
LastEditTime: 2025-06-03 14:44:13
Description: 将接口用于卷积注意力 并且和Vit结合
'''
import torch
from torch import nn
from Model.pos_embed import get_2d_sincos_pos_embed
from timm.models.vision_transformer import Block
class AdaptiveModulator(nn.Module):
    def __init__(self, M):
        super(AdaptiveModulator, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(1, M),
            nn.ReLU(),
            nn.Linear(M, M),
            nn.ReLU(),
            nn.Linear(M, M),
            nn.Sigmoid()
        )

    def forward(self, snr):
        return self.fc(snr)
class ResBlock(nn.Module):
    def __init__(self, num_filters=128):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(num_filters, num_filters//2, 1, stride=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(num_filters//2, num_filters//2, 3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(num_filters//2, num_filters, 1, stride=1)

    def forward(self, x):
        res = self.relu1(self.conv1(x))
        res = self.relu2(self.conv2(res))
        res = self.conv3(res)
        res += x
        return res

class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y
class ConvVitChannelEncoder(nn.Module):
    def __init__(self, img_size = (8, 8), input_dims = 96,
                 embed_dims = 128, depths=2, num_heads = 4, C = 256,
                 mlp_ratio=4., qkv_bias=True,
                 norm_layer=nn.LayerNorm):
        super().__init__()
        self.conv = nn.Sequential(
            ResBlock(input_dims),
            ResBlock(input_dims),
            ResBlock(input_dims),
            nn.BatchNorm2d(input_dims)
        )

        self.seblock = SEBlock(input_dims)

        self.head_list = nn.Linear(input_dims, embed_dims)
        self.pos_embed = nn.Parameter(torch.zeros(1, img_size[0]*img_size[1], embed_dims), requires_grad=False)
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(img_size[0]), cls_token=False)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        # self.layers = nn.ModuleList()
        self.blocks = nn.ModuleList([
            Block(dim=embed_dims,
                            num_heads=num_heads, 
                            mlp_ratio=mlp_ratio,
                            qkv_bias=qkv_bias,
                            norm_layer=norm_layer)
            for i in range(depths)])
        
        self.layer_num = layer_num = 7
        self.hidden_dim = int(embed_dims * 1.5)
        self.bm_list = nn.ModuleList()
        self.sm_list = nn.ModuleList()
        self.sm_list.append(nn.Linear(embed_dims, self.hidden_dim))
        for i in range(layer_num):
            if i == layer_num - 1:
                outdim = embed_dims
            else:
                outdim = self.hidden_dim
            self.bm_list.append(AdaptiveModulator(self.hidden_dim))
            self.sm_list.append(nn.Linear(self.hidden_dim, outdim))
        self.sigmoid = nn.Sigmoid()
        self.tail_list = nn.Linear(embed_dims*img_size[0]*img_size[1], C)

    def forward(self, x, importance, snr):

        trunk = self.conv(x)
        x = x + torch.mul(trunk, importance) # [B 96 8 8]
        x = self.seblock(x)
        x = x.flatten(2).permute(0, 2, 1) # B 64 96
        x = self.head_list(x)# B 64 256
        x = x + self.pos_embed
        B, L, C = x.shape
        skip1 = x
        for _, blk in enumerate(self.blocks):
            x = blk(x)
        x = x + skip1
        #ChannelModNet
        device = x.get_device()
        # device = "cpu"
        snr_cuda = torch.tensor(snr, dtype=torch.float).to(device)
        snr_batch = snr_cuda.unsqueeze(0).expand(B, -1)
        for i in range(self.layer_num):
            if i == 0:
                temp = self.sm_list[i](x.detach())
            else:
                temp = self.sm_list[i](temp)

            bm = self.bm_list[i](snr_batch).unsqueeze(1).expand(-1, L, -1)
            temp = temp * bm
        mod_val = self.sigmoid(self.sm_list[-1](temp))
        x = x * mod_val
        
        x = x.flatten(1)# B 64*256
        out = self.tail_list(x)

        return out

class ConvVitChannelDecoder(nn.Module):
    def __init__(self, img_size = (8, 8), input_dims = 256,
                 embed_dims = 128, depths=2, num_heads = 4, C = 96,
                 mlp_ratio=4., qkv_bias=True,
                 norm_layer=nn.LayerNorm):
        super().__init__()
        self.embed_dims = embed_dims
        self.img_size = img_size
        self.head_list = nn.Linear(input_dims, embed_dims*img_size[0]*img_size[1])

        self.pos_embed = nn.Parameter(torch.zeros(1, img_size[0]*img_size[1], embed_dims), requires_grad=False)
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(img_size[0]), cls_token=False)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        # self.layers = nn.ModuleList()
        self.blocks = nn.ModuleList([
            Block(dim=embed_dims,
                            num_heads=num_heads, 
                            mlp_ratio=mlp_ratio,
                            qkv_bias=qkv_bias,
                            norm_layer=norm_layer)
            for i in range(depths)])
        self.tail_list = nn.Linear(embed_dims, C)
        self.conv = nn.Sequential(
            ResBlock(C),
            ResBlock(C),
            ResBlock(C),
            nn.BatchNorm2d(C)
        )
        self.seblock = SEBlock(C)

        self.layer_num = layer_num = 7
        self.hidden_dim = int(embed_dims * 1.5)
        self.bm_list = nn.ModuleList()
        self.sm_list = nn.ModuleList()
        self.sm_list.append(nn.Linear(embed_dims, self.hidden_dim))
        for i in range(layer_num):
            if i == layer_num - 1:
                outdim = embed_dims
            else:
                outdim = self.hidden_dim
            self.bm_list.append(AdaptiveModulator(self.hidden_dim))
            self.sm_list.append(nn.Linear(self.hidden_dim, outdim))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, importance, snr):
        x = self.head_list(x) # B 64 256
        x = x.view(-1, self.img_size[0]*self.img_size[1], self.embed_dims) # B 64 256
        B,L,C = x.shape
        device = x.get_device()
        # device = "cpu"
        snr_cuda = torch.tensor(snr, dtype=torch.float).to(device)
        snr_batch = snr_cuda.unsqueeze(0).expand(B, -1)
        for i in range(self.layer_num):
            if i == 0:
                temp = self.sm_list[i](x.detach())
            else:
                temp = self.sm_list[i](temp)
            bm = self.bm_list[i](snr_batch).unsqueeze(1).expand(-1, L, -1)
            temp = temp * bm
        mod_val = self.sigmoid(self.sm_list[-1](temp))
        x = x * mod_val

        x = x + self.pos_embed
        B, L, C = x.shape
        skip1 = x
        for _, blk in enumerate(self.blocks):
            x = blk(x)
        x = x + skip1 # [B 64 256]
        x = self.tail_list(x) # [B 64 96]
        #IAN
        x = x.reshape(-1, self.img_size[0], self.img_size[1], 96).permute(0, 3, 2, 1)
        trunk = self.conv(x)
        x = x + torch.mul(trunk, importance) # [B 96 8 8]
        x = self.seblock(x)
        #----
        x = x.flatten(2).permute(0, 2, 1) # [B 64 96]
        return self.sigmoid(x)



if __name__ == "__main__":
    img = torch.randn(1, 96, 8, 8)
    importance = torch.randn(1, 96, 8, 8)
    model = ConvVitChannelEncoder()
    model2 = ConvVitChannelDecoder()
    out = model(img, importance, 1)
    out2 = model2(out, importance, 1)
    print(out.shape)
    print(out2.shape)