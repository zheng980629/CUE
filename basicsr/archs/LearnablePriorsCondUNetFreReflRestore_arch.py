from turtle import forward
from weakref import ref
import torch
from torch import nn as nn

from basicsr.archs.arch_util import ResidualBlockNoBN, Upsample, make_layer
from basicsr.utils.registry import ARCH_REGISTRY
import torch.nn.functional as F

from collections import OrderedDict

import torch
import torch.nn as nn


class BasicUnit(nn.Module):
    def __init__(self,
                 in_channels,
                 mid_channels,
                 out_channels,
                 kernel_size=3):
        super(BasicUnit, self).__init__()
        p = kernel_size//2
        self.basic_unit = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size, padding=p, bias=False),
            nn.LeakyReLU(0.2),
            nn.Conv2d(mid_channels, mid_channels, kernel_size, padding=p, bias=False),
            nn.LeakyReLU(0.2),
            nn.Conv2d(mid_channels, out_channels, kernel_size, padding=p, bias=False)
            )

    def forward(self, input):
        return self.basic_unit(input)


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

        return xout, x3


class UNetConvBlock_fre(nn.Module):
    def __init__(self, in_size, out_size, relu_slope=0.1, use_HIN=True):
        super(UNetConvBlock_fre, self).__init__()
        self.identity = nn.Conv2d(in_size, out_size, 1, 1, 0)

        self.conv_1 = nn.Conv2d(in_size, out_size, kernel_size=3, padding=1, bias=True)
        self.relu_1 = nn.LeakyReLU(relu_slope, inplace=False)
        self.conv_2 = nn.Conv2d(out_size, out_size, kernel_size=3, padding=1, bias=True)
        self.relu_2 = nn.LeakyReLU(relu_slope, inplace=False)

        if use_HIN:
            self.norm = nn.InstanceNorm2d(out_size // 2, affine=True)
        self.use_HIN = use_HIN

    def forward(self, x):
        out = self.conv_1(x)
        if self.use_HIN:
            out_1, out_2 = torch.chunk(out, 2, dim=1)
            out = torch.cat([self.norm(out_1), out_2], dim=1)
        out = self.relu_1(out)
        out = self.relu_2(self.conv_2(out))
        out += self.identity(x)

        return out



class InvBlock(nn.Module):
    def __init__(self, channel_num, channel_split_num, clamp=0.8):
        super(InvBlock, self).__init__()
        # channel_num: 3
        # channel_split_num: 1

        self.split_len1 = channel_split_num  # 1
        self.split_len2 = channel_num - channel_split_num  # 2

        self.clamp = clamp

        self.F = UNetConvBlock_fre(self.split_len2, self.split_len1)
        self.G = UNetConvBlock_fre(self.split_len1, self.split_len2)
        self.H = UNetConvBlock_fre(self.split_len1, self.split_len2)

        self.flow_permutation = lambda z, logdet, rev: self.invconv(z, logdet, rev)

    def forward(self, x):
        # split to 1 channel and 2 channel.
        x1, x2 = (x.narrow(1, 0, self.split_len1), x.narrow(1, self.split_len1, self.split_len2))

        y1 = x1 + self.F(x2)  # 1 channel
        self.s = self.clamp * (torch.sigmoid(self.H(y1)) * 2 - 1)
        y2 = x2.mul(torch.exp(self.s)) + self.G(y1)  # 2 channel
        out = torch.cat((y1, y2), 1)

        return out



class SpaBlock(nn.Module):
    def __init__(self, nc):
        super(SpaBlock, self).__init__()
        self.block = InvBlock(nc,nc//2)

    def forward(self, x):
        return x+self.block(x)


class FreBlockSpa(nn.Module):
    def __init__(self, nc):
        super(FreBlockSpa, self).__init__()
        self.processreal = nn.Sequential(
            nn.Conv2d(nc,nc,kernel_size=3,padding=1,stride=1,groups=nc),
            nn.LeakyReLU(0.1,inplace=True),
            nn.Conv2d(nc,nc,kernel_size=3,padding=1,stride=1,groups=nc))
        self.processimag = nn.Sequential(
            nn.Conv2d(nc, nc, kernel_size=3, padding=1, stride=1, groups=nc),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(nc, nc, kernel_size=3, padding=1, stride=1, groups=nc))

    def forward(self,x):
        real = self.processreal(x.real)
        imag = self.processimag(x.imag)
        x_out = torch.complex(real, imag)

        return x_out


class FreBlockCha(nn.Module):
    def __init__(self, nc):
        super(FreBlockCha, self).__init__()
        self.processreal = nn.Sequential(
            nn.Conv2d(nc,nc,kernel_size=1,padding=0,stride=1),
            nn.LeakyReLU(0.1,inplace=True),
            nn.Conv2d(nc,nc,kernel_size=1,padding=0,stride=1))
        self.processimag = nn.Sequential(
            nn.Conv2d(nc, nc, kernel_size=1, padding=0, stride=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(nc, nc, kernel_size=1, padding=0, stride=1))

    def forward(self,x):
        real = self.processreal(x.real)
        imag = self.processimag(x.imag)
        x_out = torch.complex(real, imag)

        return x_out


class SpatialFuse(nn.Module):
    def __init__(self, in_nc):
        super(SpatialFuse,self).__init__()
        # self.fpre = nn.Conv2d(in_nc, in_nc, 1, 1, 0)
        self.spatial_process = SpaBlock(in_nc)
        self.frequency_process = FreBlockSpa(in_nc)
        self.frequency_spatial = nn.Conv2d(in_nc,in_nc,3,1,1)
        self.cat = nn.Conv2d(2*in_nc,in_nc,3,1,1)


    def forward(self, x):
        xori = x
        _, _, H, W = x.shape
        x_freq = torch.fft.rfft2(x, norm='backward')
        x = self.spatial_process(x)
        x_freq = self.frequency_process(x_freq)
        x_freq_spatial = torch.fft.irfft2(x_freq, s=(H, W), norm='backward')
        x_freq_spatial = self.frequency_spatial(x_freq_spatial)
        xcat = torch.cat([x,x_freq_spatial],1)
        x_out = self.cat(xcat)

        return x_out+xori


class ChannelFuse(nn.Module):
    def __init__(self, in_nc):
        super(ChannelFuse,self).__init__()
        # self.fpre = nn.Conv2d(in_nc, in_nc, 1, 1, 0)
        self.spatial_process = SpaBlock(in_nc)
        self.frequency_process = FreBlockCha(in_nc)
        self.frequency_spatial = nn.Conv2d(in_nc,in_nc,1,1,0)
        self.cat = nn.Conv2d(2*in_nc,in_nc,1,1,0)


    def forward(self, x):
        xori = x
        _, _, H, W = x.shape
        x_freq = torch.fft.rfft2(x, norm='backward')
        x = self.spatial_process(x)
        x_freq = self.frequency_process(x_freq)
        x_freq_spatial = torch.fft.irfft2(x_freq, s=(H, W), norm='backward')
        x_freq_spatial = self.frequency_spatial(x_freq_spatial)
        xcat = torch.cat([x,x_freq_spatial],1)
        x_out = self.cat(xcat)

        return x_out+xori


class ProcessBlock(nn.Module):
    def __init__(self, nc):
        super(ProcessBlock, self).__init__()
        self.spa = SpatialFuse(nc)
        self.cha = ChannelFuse(nc)

    def forward(self,x):
        x = self.spa(x)
        x = self.cha(x)

        return x


class ProcessNet(nn.Module):
    def __init__(self, nc):
        super(ProcessNet,self).__init__()
        self.conv0 = nn.Conv2d(nc, nc, 3, 1, 1)
        self.conv1 = ProcessBlock(nc)
        self.downsample1 = nn.Conv2d(nc, nc * 2, stride=2, kernel_size=2, padding=0)
        self.conv2 = ProcessBlock(nc * 2)
        self.downsample2 = nn.Conv2d(nc * 2, nc * 3, stride=2, kernel_size=2, padding=0)
        self.conv3 = ProcessBlock(nc * 3)
        self.up1 = nn.ConvTranspose2d(nc * 5, nc * 2, 1, 1)
        self.conv4 = ProcessBlock(nc * 2)
        self.up2 = nn.ConvTranspose2d(nc * 3, nc * 1, 1, 1)
        self.conv5 = ProcessBlock(nc)
        self.convout = nn.Conv2d(nc, nc, 3, 1, 1)

    def forward(self, x):
        x = self.conv0(x)
        x01 = self.conv1(x)
        x1 = self.downsample1(x01)
        x12 = self.conv2(x1)
        x2 = self.downsample2(x12)
        x3 = self.conv3(x2)
        x34 = self.up1(torch.cat([F.interpolate(x3, size=(x12.size()[2], x12.size()[3]), mode='bilinear'), x12], 1))
        x4 = self.conv4(x34)
        x4 = self.up2(torch.cat([F.interpolate(x4, size=(x01.size()[2], x01.size()[3]), mode='bilinear'), x01], 1))
        x5 = self.conv5(x4)
        xout = self.convout(x5)

        return xout


class InteractNet(nn.Module):
    def __init__(self, inchannel, nc, outchannel):
        super(InteractNet,self).__init__()
        self.extract =  nn.Conv2d(inchannel, nc,1,1,0)
        self.process = ProcessNet(nc)
        self.recons = nn.Conv2d(nc, outchannel, 1, 1, 0)

    def forward(self, x):
        x_f = self.extract(x)
        x_f = self.process(x_f)+x_f
        y = self.recons(x_f)

        return y


class UNet_adjustment(nn.Module):
    def __init__(self, in_channels=4, channels=6, out_channels=1):
        super(UNet_adjustment,self).__init__()
        self.convpre = nn.Conv2d(in_channels, channels, 3, 1, 1)
        self.conv1 = UNetConvBlock_fre(channels, channels)
        self.down1 = nn.Conv2d(channels, 2*channels, stride=2, kernel_size=2)
        self.conv2 = UNetConvBlock_fre(2*channels, 2*channels)
        self.down2 = nn.Conv2d(2*channels, 4*channels, stride=2, kernel_size=2)
        self.conv3 = UNetConvBlock_fre(4*channels, 4*channels)

        self.Global = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(4 * channels, 4 * channels, 1, 1, 0),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Conv2d(4 * channels, 4 * channels, 1, 1, 0))
        self.context_g = UNetConvBlock_fre(8 * channels, 4 * channels)

        self.context2 = UNetConvBlock_fre(2 * channels, 2 * channels)
        self.context1 = UNetConvBlock_fre(channels, channels)

        self.merge2 = nn.Sequential(nn.Conv2d(6*channels,4*channels,1,1,0),
                                    CALayer(4*channels,4),
                                    nn.Conv2d(4*channels,2*channels,3,1,1)
                                    )
        self.merge1 = nn.Sequential(nn.Conv2d(3*channels,channels,1,1,0),
                                    CALayer(channels,2),
                                    nn.Conv2d(channels,channels,3,1,1)
                                    )

        self.conv_last = nn.Conv2d(channels,out_channels,3,1,1)
        self.relu = nn.ReLU()


    def forward(self, x, ratio):
        x = torch.cat((x, ratio), 1)
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

        return self.relu(xout)


class IlluminationBlock(nn.Module):
    def __init__(self, illu_channel, mid_channels, kernel_size, unet_channel=24):
        super(IlluminationBlock, self).__init__()

        self.L_learnedPrior = UNet_BilateralFilter_mask(in_channels=1, channels=6, out_channels=1)
        self.L_learnedPrior.load_state_dict(torch.load('illuminationPrior.pth')['params'])

        self.modulation_mul = nn.Sequential(
            nn.Conv2d(unet_channel, mid_channels, 3, padding=1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(mid_channels, mid_channels, 3, padding=1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(mid_channels, illu_channel, 3, padding=1, bias=False),
            nn.Sigmoid()
            )

        self.modulation_add = nn.Sequential(
            nn.Conv2d(unet_channel, mid_channels, 3, padding=1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(mid_channels, mid_channels, 3, padding=1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(mid_channels, illu_channel, 3, padding=1, bias=False),
            nn.Sigmoid()
            )

    def forward(self, low_light, illu, noise, refl, alpha, mu):
        L_prior, L_pripr_feat = self.L_learnedPrior(illu)

        L_cat = torch.cat([illu, illu, illu], 1)
        identity = torch.ones_like(L_cat)
        L_hat = (identity - alpha * refl * refl) * illu - alpha * refl * (noise - low_light)
        illu = torch.mean(L_hat, 1).unsqueeze(1)

        L_pripr_feat = F.interpolate(L_pripr_feat, size=illu.shape[-2:], mode='bilinear', align_corners=True)
        # illu = illu * self.modulation_mul(L_pripr_feat) + self.modulation_add(L_pripr_feat)
        illu = illu + self.modulation_add(L_pripr_feat)

        return illu, L_hat


class ReflectanceBlock(nn.Module):
    def __init__(self, refl_channel, mid_channels, kernel_size):
        super(ReflectanceBlock, self).__init__()
        self.prox = BasicUnit(refl_channel, mid_channels, refl_channel, kernel_size)

    def forward(self, low_light, illu, noise, refl, beta, mu):

        identity = torch.ones_like(illu)

        refl_hat = (identity - beta * illu * illu) * refl - beta * illu * (noise - low_light)
        refl = self.prox(refl_hat) + refl_hat
        # refl = refl_hat

        return refl


class NoiseBlock(nn.Module):
    def __init__(self, noise_channel, mid_channels, kernel_size):
        super(NoiseBlock, self).__init__()
        self.prox = BasicUnit(noise_channel, mid_channels, noise_channel, kernel_size)

    def shrink(self, x, r):
        zeros = torch.zeros_like(x)
        z = torch.sign(x) * torch.max(torch.abs(x) - r, zeros)
        return z

    def forward(self, low_light, illu, refl, mu):
        illu_cat = torch.cat([illu, illu, illu], 1)
        noise_hat = self.shrink(low_light - refl * illu_cat, 1 / mu)
        noise = self.prox(noise_hat) + noise_hat
        # noise = noise_hat

        return noise



@ARCH_REGISTRY.register()
class LearnablePriorEnhanceNet(nn.Module):
    def __init__(self, stage, illu_channel, refl_channel, noise_channel, num_feat, ratio, alpha=0.001, beta=0.001, mu=0.1):
        super(LearnablePriorEnhanceNet, self).__init__()
        # loading decomposition model
        self.model_illu = IlluminationBlock(illu_channel, num_feat, 1)
        self.model_refl = ReflectanceBlock(refl_channel, num_feat, 1)
        self.model_noise = NoiseBlock(noise_channel, num_feat, 1)
        self.adjustIllu_model = InteractNet(inchannel=2, nc=8, outchannel=1)
        self.restoraRefl_model = InteractNet(inchannel=3, nc=8, outchannel=3)

        self.alpha = nn.Parameter(torch.tensor([alpha]), False)
        self.beta = nn.Parameter(torch.tensor([beta]), False)
        self.mu = nn.Parameter(torch.tensor([mu]))
        self.stage = stage
        self.ratio = ratio

    def unfolding(self, input_low_img):
        for t in range(self.stage):
            if t == 0: # initialize
                illu = torch.max(input_low_img, 1)[0].unsqueeze(1)
                refl = input_low_img / (illu + 1e-8)
                noise = torch.zeros_like(input_low_img).cuda()
            else: # update
                illu, L_prior_cond = self.model_illu(input_low_img, illu, noise, refl, self.alpha, self.mu)
                refl = self.model_refl(input_low_img, illu, noise, refl, self.beta, self.mu)
                noise = self.model_noise(input_low_img, illu, refl, self.mu)
        return refl, illu, noise, L_prior_cond

    def illumination_adjust(self, L, ratio):
        ratio = torch.ones(L.shape).cuda() * self.ratio
        input = torch.cat([L, ratio], 1)
        return self.adjustIllu_model(input)

    def forward(self, input_low_img):
        R, L, noise, L_pripr_cond = self.unfolding(input_low_img)
        High_L = self.illumination_adjust(L, self.ratio)
        restored_R = self.restoraRefl_model(R)
        I_enhance = High_L * restored_R

        return I_enhance, High_L, L, restored_R, R, noise, L_pripr_cond