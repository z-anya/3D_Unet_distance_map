from __future__ import print_function
import argparse
import pdb
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import einops


# 未测试
class SEBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SEBlock, self).__init__()
        self.gap = nn.AdaptiveAvgPool3d(1)
        conv = nn.Conv3d

        self.conv1 = conv(in_channels, out_channels, 1)
        self.conv2 = conv(in_channels, out_channels, 1)

        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        inpu = x
        x = self.gap(x)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.sigmoid(x)

        return inpu * x


# 未测试
class DenseBlock(nn.Module):
    def __init__(self, channels, conv_num):
        super(DenseBlock, self).__init__()
        self.conv_num = conv_num
        conv = nn.Conv3d

        self.relu = nn.ReLU()
        self.conv_list = []
        self.bottle_conv_list = []
        for i in conv_num:
            self.bottle_conv_list.append(conv(channels * (i + 1), channels * 4, 1))
            self.conv_list.append(conv(channels * 4, channels, 3, padding=1))

    def forward(self, x):

        res_x = []
        res_x.append(x)

        for i in self.conv_num:
            inputs = torch.cat(res_x, dim=1)
            x = self.bottle_conv_list[i](inputs)
            x = self.relu(x)
            x = self.conv_list[i](x)
            x = self.relu(x)
            res_x.append(x)

        return x


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv1 = nn.Conv3d(in_channels, out_channels, 3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(out_channels, out_channels, 3, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm3d(out_channels)

        if in_channels != out_channels:
            self.res_conv = nn.Conv3d(in_channels, out_channels, 1, stride=stride)

    def forward(self, x):
        if self.in_channels != self.out_channels:
            res = self.res_conv(x)
        else:
            res = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)

        out = x + res
        out = self.relu(out)

        return out


# Transformer Decoder结构
class AttnBlock(nn.Module):
    def __init__(self, config, in_channels, *args, **kwargs):
        super().__init__(*args, **kwargs)
        conv = partial(nn.Conv3d, in_channels=in_channels, out_channels=in_channels, kernel_size=1, padding=0)
        self.norm = nn.GroupNorm(num_groups=in_channels, num_channels=in_channels)
        self.q, self.k, self.v = conv(), conv(), conv()
        self.proj_out = conv()
        # self.rope =

    def forward(self, x, y):
        residual = x
        q = self.q(y)
        k, v = self.k(x), self.v(x)
        b, c, z, h, w = q.shape

        # q (b, n, d)
        q = einops.rearrange(q, 'b c z h w -> b (z h w) c')
        # k.T (b, d, n)
        k = einops.rearrange(k, 'b c z h w -> b c (z h w)')
        v = einops.rearrange(v, 'b c z h w -> b (z h w) c')

        #compute attn matrix
        attn_weight = torch.bmm(q, k)
        attn_weight = attn_weight * (int(c) ** -0.5)
        attn_weight = F.softmax(attn_weight, dim=-1)
        y = torch.bmm(attn_weight, v)
        y = einops.rearrange(y, 'b (z h w) c -> b c z h w', z=z, h=h, w=w)
        y += residual

        # ffw
        y = self.proj_out(y)
        y += residual

        return y

