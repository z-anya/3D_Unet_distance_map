import argparse
import pdb
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from .nn.module import AttnBlock


class UnitedNet(nn.Module):
    def __init__(self, config):
        super(UnitedNet, self).__init__()
        # segment unet
        self.seg_unet = ZyxUNet(config, True)
        # distance map unet
        self.dm_unet = ZyxUNet(config, False)
        # center line unet
        self.cl_unet = ZyxUNet(config, False)

    def forward(self, x):
        # encoder
        dm_maps, dm_mid = self.dm_unet.encoder(x)
        cl_maps, cl_mid = self.cl_unet.encoder(x)
        seg_maps, seg_mid = self.seg_unet.encoder(x, cl_maps)
        # attn block
        seg_mid = self.seg_unet.attn_block(seg_mid, dm_mid)

        # decoder
        seg_output = self.seg_unet.decoder(seg_mid, seg_maps)
        dm_output = self.dm_unet.decoder(dm_mid, dm_maps)
        cl_output = self.cl_unet.decoder(cl_mid, cl_maps)

        return seg_output, dm_output, cl_output


class ZyxUNet(nn.Module):
    def __init__(self, config, has_attn=False):
        super(ZyxUNet, self).__init__()
        self.encoder = EncoderZyxUNet(config)
        self.decoder = DecoderZyxUNet(config)
        self.has_attn = has_attn
        if self.has_attn:
            self.attn_block = AttnBlock(config, self.encoder.last_channel)
    # def forward(self, x, y=None):
    #     x = self.encoder(x)
    #     if self.has_attn:
    #         assert y is not None
    #         x = self.attn_block(x, y)
    #     x = self.decoder(x)
    #     return x

class EncoderZyxUNet(nn.Module):
    def __init__(self, config, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 模型初始通道数
        start_channel = config.encoder_start_channel

        dim_in_and_out, self.last_channel = get_layer_dim(config.encoder_layer_num, config.encoder_start_channel, is_decoder=False)

        # 将输出首先转换到模型初始通道数
        self.conv_in = nn.Conv3d(config.init_channel, start_channel, kernel_size=3, padding=1, stride=1)

        self.down_blocks = nn.ModuleList()
        for (dim_in, dim_out) in dim_in_and_out:
            # print(f"dim_in: {dim_in}, dim_out: {dim_out}")
            self.down_blocks.append(nn.Sequential(
                nn.Conv3d(dim_in, dim_out, kernel_size=3, padding=1, stride=1),
                nn.GroupNorm(start_channel, dim_out),
                nn.LeakyReLU(0.1),
                MaxPool3dWrapper(kernel_size=2, stride=2) \
                    if config.downsample_type == 'max_pool' \
                    else nn.Conv3d(dim_out, dim_out, kernel_size=4, padding=1, stride=2)
            ))
    def forward(self, x, cl_maps=None):
        # if cl_maps is not None, that means this encoder is from seg_unet. Each layer will be addded the output of each cl encoder layers
        x = self.conv_in(x)
        maps = []
        if cl_maps is not None:
            for block, cl_map in zip(self.down_blocks, cl_maps):
                x = block(x)
                # 为下一次提前加上x
                x += cl_map
                maps.append(x)
        else:
            for block in self.down_blocks:
                x = block(x)
                maps.append(x)
        return maps, x

class DecoderZyxUNet(nn.Module):
    def __init__(self, config,  *args, **kwargs):
        super().__init__(*args, **kwargs)
        start_channel = config.encoder_start_channel
        dim_in_and_out, last_channel = get_layer_dim(config.encoder_layer_num, config.encoder_start_channel, is_decoder=True)
        self.up_blocks = nn.ModuleList()
        self.conv_out = nn.Conv3d(last_channel, config.init_channel, kernel_size=3, padding=1, stride=1)
        for (dim_in, dim_out) in dim_in_and_out:
            self.up_blocks.append(nn.Sequential(
                nn.Conv3d(dim_in, dim_out, kernel_size=3, padding=1, stride=1),
                nn.GroupNorm(start_channel, dim_out),
                nn.LeakyReLU(0.1),
                InterpolateWrapper(scale_factor=(2, 2, 2), mode='trilinear') \
                    if config.upsample_type == 'interpolate' \
                    else nn.ConvTranspose3d(dim_out, dim_out, kernel_size=4, padding=1, stride=2)
            ))

    def forward(self, x, maps):
        assert len(self.up_blocks) == len(maps)
        # shapes = [item.shape for item in maps]
        # pdb.set_trace()
        maps.reverse()
        for block, map_feature in zip(self.up_blocks, maps):
            # print(f"x: {x.shape}, map_feature: {map_feature.shape}")
            x = block(x + map_feature)
        x = self.conv_out(x)
        return x

class InterpolateWrapper(nn.Module):
    def __init__(self, scale_factor, mode):
        super(InterpolateWrapper, self).__init__()
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        return F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)


class MaxPool3dWrapper(nn.Module):
    def __init__(self, kernel_size=2, stride=2):
        super(MaxPool3dWrapper, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, x):
        return F.max_pool3d(x, kernel_size=self.kernel_size, stride=self.stride)


def get_layer_dim(encoder_layer_num, encoder_start_channel, is_decoder=False):
    layer_num = encoder_layer_num
    # 模型初始通道数
    start_channel = encoder_start_channel
    # 计算每一层的输入输出维度应该是多少, 方便后面构建卷积层
    module_mult = list(map(lambda x: 2 ** x, range(layer_num)))  # 每次通道上升的倍数
    dims = [start_channel * item for item in module_mult]  # 由初始通道数乘倍数计算每层卷积的通道数
    if is_decoder:
        dims.reverse()
    dim_in_and_out = zip(dims[:-1], dims[1:])
    return dim_in_and_out, dims[-1]

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--upsample_type', type=str, default='')
    parser.add_argument('--downsample_type', type=str, default='')
    parser.add_argument('--init_channel', type=int, default=1)
    parser.add_argument('--encoder_start_channel', type=int, default=32)
    parser.add_argument('--encoder_layer_num', type=int, default=6)
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    model = ZyxUNet(args)
    print(model)