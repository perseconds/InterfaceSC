'''
date 2025/2/22
author:kwz
net Model
Modify:

'''
import math
import torch
from torch import nn
from Model.BitImportance import BSC
from Model.Encoder import *
from Model.Decoder import *
from Model.channel import Channel
from Model.Conv_Vit import ConvVitChannelDecoder, ConvVitChannelEncoder




import torch
from torch import Tensor

#----------------------模型--------------------------------
class net(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.H = self.W = 0
        self.downsample = config.downsample
        self.encoder = SourceEncoder(**config.encoder_kwargs)
        self.decoder = SourceDecoder(**config.decoder_kwargs)
        L = (config.encoder_kwargs['img_size'][0] // 4)**2
        self.BSCchannel = BSC((L, config.encoder_kwargs['C']))
    def forward(self, x, is_train=True):
        B, _, H, W = x.shape

        if H != self.H or W != self.W:
            self.encoder.update_resolution(H, W)
            self.decoder.update_resolution(H // (2 ** self.downsample), W // (2 ** self.downsample))
            self.H = H
            self.W = W

        binary_prob = self.encoder(x)
        noise_prob = self.BSCchannel() # L N 会自动广播
        binary_code = torch.round(binary_prob)#STE
        if is_train:
            return self._forward_train(binary_prob, binary_code, noise_prob)
        else:
            return self._forward_eval(binary_code, noise_prob)
    def _forward_train(self, binary_prob, binary_code, noise_prob):
        noise_binary_prob = binary_prob - (2 * binary_prob * noise_prob) + noise_prob
        #即 binary_prob*(1 - noise_prob) + (1 - binary_prob)*noise_prob
       
        noise_binary_code = noise_binary_prob + (torch.bernoulli(noise_binary_prob) - noise_binary_prob).detach()# 利用STE的做法，直接反传梯度
        out = self.decoder(noise_binary_code)
        return out.clamp(-1, 1), binary_code, noise_prob#训练时binarycode也用不到 随便返回一个
    def _forward_eval(self, binary_code, noise_prob):
        with torch.no_grad():
            out = self.decoder(binary_code)
            return out.clamp(-1, 1), binary_code, noise_prob
            
class ChannelCode(net):
    def __init__(self, config, args):
        super().__init__(config)

        self.chan_encoder = ConvVitChannelEncoder(**config.chan_en)
        self.chan_decoder = ConvVitChannelDecoder(**config.chan_de)
        self.channel = Channel(args.channel)
    def feature_pass_channel(self, feature, chan_param, avg_pwr=False):
        noisy_feature = self.channel.forward(feature, chan_param, avg_pwr)
        return noisy_feature
    def forward(self, x, snr):
        _, binary_code, noise_prob =  super().forward(x, is_train=False)
        #----------------信道编码--------------------
        B, L, N = binary_code.shape
        noise_prob = noise_prob.expand_as(binary_code)
        H = int(math.sqrt(L))
        W = int(math.sqrt(L))
        binary_code_reshape = binary_code.reshape(B, H, W, N).permute(0, 3, 1, 2)# B C H W
        noise_prob = noise_prob.reshape(B, H, W, N).permute(0, 3, 1, 2) # B C H W
        
        y = self.chan_encoder(binary_code_reshape, 1 - 2 * noise_prob, snr) # B * K
        # print(y)
        CBR = y.numel() / 2 / x.numel()
        noisy_feature = self.feature_pass_channel(y, snr)
        # noisy_feature = y
        # print(y.shape)
    
        y_hat = self.chan_decoder(noisy_feature,  1 - 2 * noise_prob, snr) # B L C
        y_quan = torch.round(y_hat) - y_hat.detach() + y_hat#STE
        recon_x = self.decoder(y_quan)
        return recon_x, CBR


if __name__ == "__main__":

    img = torch.randn((16, 3, 32 ,32))
    model = ChannelCode()
    out, _ = model(img, 1)
    # loss = torch.mean(out ** 2)
    # loss.backward()

    print(out.shape)
    print(model.chan_decoder.map.weight.grad)
    # print(binary_code.size())
    # print(noise_prob.shape)
    

