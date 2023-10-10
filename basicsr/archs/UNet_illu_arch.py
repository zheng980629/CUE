import torch
from torch import nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import math
import numpy as np
import os
from math import exp

from basicsr.archs.arch_util import ResidualBlockNoBN, Upsample, make_layer, ConvLReLUNoBN, upsample_and_concat
from basicsr.utils.registry import ARCH_REGISTRY


class UNetConvBlock(nn.Module):
    def __init__(self, in_size, out_size, relu_slope=0.1):
        super(UNetConvBlock, self).__init__()
        self.identity = nn.Conv2d(in_size, out_size, 1, 1, 0)
        self.conv_1 = nn.Conv2d(in_size, out_size, kernel_size=3, padding=1, bias=True)
        self.relu_1 = nn.LeakyReLU(relu_slope, inplace=False)
        self.conv_2 = nn.Conv2d(out_size, out_size, kernel_size=3, padding=1, bias=True)
        self.relu_2 = nn.LeakyReLU(relu_slope, inplace=False)

    def forward(self, x):
        out = self.conv_1(x)
        out = self.relu_1(out)
        out = self.relu_2(self.conv_2(out))
        out += self.identity(x)

        return out


class CALayer(nn.Module):
    def __init__(self, channel, reduction):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )
        self.process = nn.Sequential(
            nn.Conv2d(channel, channel, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(channel, channel, 3, stride=1, padding=1)
        )

    def forward(self, x):
        y = self.process(x)
        y = self.avg_pool(y)
        z = self.conv_du(y)
        return z * y + x


def initialize_weights_xavier(net_l, scale=1):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                init.xavier_normal_(m.weight)
                m.weight.data *= scale  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.xavier_normal_(m.weight)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)

@ARCH_REGISTRY.register()
class UNet_BilateralFilter_mask(nn.Module):
    def __init__(self, in_channels=4, channels=6, out_channels=1):
        super(UNet_BilateralFilter_mask,self).__init__()
        self.convpre = nn.Conv2d(in_channels, channels, 3, 1, 1)
        self.conv1 = UNetConvBlock(channels, channels)
        self.down1 = nn.Conv2d(channels, 2*channels, stride=2, kernel_size=2)
        self.conv2 = UNetConvBlock(2*channels, 2*channels)
        self.down2 = nn.Conv2d(2*channels, 4*channels, stride=2, kernel_size=2)
        self.conv3 = UNetConvBlock(4*channels, 4*channels)

        self.Global = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(4 * channels, 4 * channels, 1, 1, 0),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Conv2d(4 * channels, 4 * channels, 1, 1, 0))
        self.context_g = UNetConvBlock(8 * channels, 4 * channels)

        self.context2 = UNetConvBlock(2 * channels, 2 * channels)
        self.context1 = UNetConvBlock(channels, channels)

        self.merge2 = nn.Sequential(nn.Conv2d(6*channels,4*channels,1,1,0),
                                    CALayer(4*channels,4),
                                    nn.Conv2d(4*channels,2*channels,3,1,1)
                                    )
        self.merge1 = nn.Sequential(nn.Conv2d(3*channels,channels,1,1,0),
                                    CALayer(channels,2),
                                    nn.Conv2d(channels,channels,3,1,1)
                                    )

        self.conv_last = nn.Conv2d(channels,out_channels,3,1,1)


    def forward(self, x):
        x1 = self.conv1(self.convpre(x))
        x2 = self.conv2(self.down1(x1))
        x3 = self.conv3(self.down2(x2))

        x_global = self.Global(x3)
        _,_,h,w = x3.size()
        x_global = x_global.repeat(1,1,h,w)
        x3 = self.context_g(torch.cat([x_global,x3],1))

        x3 = F.interpolate(x3, scale_factor=2, mode='bilinear')
        x2 = self.context2(self.merge2(torch.cat([x2, x3], 1)))

        x2 = F.interpolate(x2, scale_factor=2, mode='bilinear')
        x1 = self.context1(self.merge1(torch.cat([x1, x2], 1)))

        xout = self.conv_last(x1)

        return xout

