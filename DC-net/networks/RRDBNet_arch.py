import torch
import torch.nn as nn
import torch.nn.functional as F
from .arch_util import LayerNorm2d
import numpy as _np
import einops
from typing import Any, Sequence, Tuple
import numpy as np

from .max_attention import multi_channel_attention,multi_channel,mutil_sptical_attention

from .arch_util import LayerNorm2d
from typing import Any, Sequence, Tuple
#################################################
#  空间分解（1）并联
#################################################

class Densenet(nn.Module):
    def __init__(self, channels_, output):
        super(Densenet, self).__init__()
        self.conv1 = CSDN_Tem(channels_,64)
        self.conv2 = CSDN_Tem(64,64)
        self.conv3 = CSDN_Tem(128, 64)
        self.conv4 = CSDN_Tem(128, 64)
        self.conv5 =CSDN_Tem(256, output)

        self.conv = nn.Conv2d(in_channels=channels_, out_channels=output, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x1 = F.relu(self.conv1(x))
        x2 = F.relu(self.conv2(x1))
        cat1 = torch.cat((x1, x2), 1)
        x3 = F.relu(self.conv3(cat1))
        cat2 = torch.cat((x2, x3), 1)
        x4 = F.relu(self.conv4(cat2))
        cat3 = torch.cat((x1, x2, x3, x4), 1)
        output = F.relu(self.conv5(cat3))
        # output = self.conv(x)
        return output
class SpatialAttention(nn.Module):
    def __init__(self, chns, factor):
        super(SpatialAttention, self).__init__()
        self.spatial_pool = nn.Sequential(
            nn.Conv2d(chns, chns // factor, 1, 1, 0),
            nn.LeakyReLU(),
            nn.Conv2d(chns // factor, 1, 1, 1, 0),
            nn.Sigmoid()
        )

    def forward(self, x):
        # X @ B,C,H,W
        # map @B,1,H,W
        spatial_map = self.spatial_pool(x)
        return x * spatial_map


def block_images_einops(x, patch_size):
    """Image to patches."""
    batch, channels, height, width = x.shape
    x = x.permute(0, 2, 3, 1)
    grid_height = height // patch_size[0]
    grid_width = width // patch_size[1]

    x = einops.rearrange(
        x, "n (gh fh) (gw fw) c -> n (gh gw) (fh fw) c",
        gh=grid_height, gw=grid_width, fh=patch_size[0], fw=patch_size[1])
    return x.permute(0, 3, 1, 2)


def unblock_images_einops(x, grid_size, patch_size):
    """patches to images."""
    x = x.permute(0, 2, 3, 1)
    x = einops.rearrange(
        x, "n (gh gw) (fh fw) c -> n (gh fh) (gw fw) c",
        gh=grid_size[0], gw=grid_size[1], fh=patch_size[0], fw=patch_size[1])
    return x.permute(0, 3, 1, 2)


class GridGatingUnit(nn.Module):
    """A SpatialGatingUnit as defined in the gMLP paper.

    The 'spatial' dim is defined as the second last.
    If applied on other dims, you should swapaxes first.
    """

    def __init__(self, h_size, input ,bias=True):
        super().__init__()
        self.h_size = h_size
        self.bias = bias
        self.input = input
        self.linear = nn.Linear(self.h_size, self.h_size, bias=self.bias)
        self.norm = LayerNorm2d(self.input)
    def forward(self, x):
        u, v = x.chunk(2, axis=1)

        v = self.norm(v)
        v = torch.transpose(v, -1, -2)
        v = self.linear(v)
        v = torch.transpose(v, -1, -2)
        return u * (v + 1.)



class GridGmlpLayer(nn.Module):
    """Grid gMLP layer that performs global mixing of tokens."""

    def __init__(self, in_channel, grid_size: Sequence[int], bias=True, factor=2, dropout_rate=0.0):
        super().__init__()
        self.in_channel = in_channel
        self.grid_size = grid_size
        self.bias = bias
        self.factor = factor
        self.dropout_rate = dropout_rate
        self.linear1 = nn.Linear(self.in_channel, self.in_channel, bias=self.bias)
        self.linear2 = nn.Linear(self.in_channel, self.in_channel, bias=self.bias)
        self.gridgatingunit = SpatialAttention(self.in_channel, 4)
        self.norm = LayerNorm2d(self.in_channel)

    def forward(self, x, deterministic=True):
        # step1: spatial split
        n, num_channels, h, w = x.shape
        gh, gw = self.grid_size
        fh, fw = h // gh, w // gw
        x = block_images_einops(x, patch_size=(fh, fw))
        # step2: gMLP1: Global (grid) mixing part, provides global grid communication.
        _n, _num_channels, _h, _w = x.shape
        y = self.norm(x)

        y = F.gelu(y)

        y = self.gridgatingunit(y)          # spatial attention

        y = F.dropout(y, self.dropout_rate, deterministic)
        x = x + y
        # step3: figure reverse
        x = unblock_images_einops(x, grid_size=(gh, gw), patch_size=(fh, fw))
        return x


class BlockGatingUnit(nn.Module):
    """A SpatialGatingUnit as defined in the gMLP paper.

    The 'spatial' dim is defined as the **second last**.
    If applied on other dims, you should swapaxes first.
    """

    def __init__(self, w_size, input ,bias=True):
        super().__init__()
        self.w_size = w_size
        self.bias = bias
        self.input = input
        self.linear = nn.Linear(self.w_size, self.w_size, bias=self.bias)
        self.norm = LayerNorm2d(self.input)
    def forward(self, x):
        u, v = x.chunk(2, axis=1)
        v = self.norm(v)
        v = self.linear(v)
        return u * (v + 1.)


class BlockGmlpLayer(nn.Module):
    """Block gMLP layer that performs local mixing of tokens."""

    def __init__(self, in_channel, block_size, bias=True, factor=2, dropout_rate=0.0):
        super().__init__()
        self.in_channel = in_channel
        self.block_size = block_size
        self.factor = factor
        self.dropout_rate = dropout_rate
        self.bias = bias
        self.linear1 = nn.Linear(self.in_channel, self.in_channel * self.factor, bias=self.bias)
        # self.linear2 = nn.Linear(self.in_channel, self.in_channel, bias=self.bias)
        # self.blockgatingunit = BlockGatingUnit(self.block_size[0] * self.block_size[1],self.in_channel)
        self.blockgatingunit = SpatialAttention(self.in_channel, 4)
        self.linear2 = nn.Linear(self.in_channel*2, self.in_channel, bias=self.bias)
        self.norm = LayerNorm2d(self.in_channel)

    def forward(self, x, deterministic=True):
        # step1: spatial split
        n, num_channels, h, w = x.shape
        fh, fw = self.block_size
        gh, gw = h // fh, w // fw
        x = block_images_einops(x, patch_size=(fh, fw))
        # step2: attention
        y = self.norm(x)

        y = F.gelu(y)

        y = self.blockgatingunit(y)

        y = F.dropout(y, self.dropout_rate, deterministic)
        x = x + y
        # step3: ublock
        x = unblock_images_einops(x, grid_size=(gh, gw), patch_size=(fh, fw))
        return x


class ResidualSplitHeadMultiAxisGmlpLayer(nn.Module):  # The multi-axis gated MLP block.
    def __init__(self, in_channel, block_size=[2, 2], grid_size=[2, 2], block_gmlp_factor=2, grid_gmlp_factor=2, input_proj_factor=2,
                 bias=True, dropout_rate=0.3):
        super().__init__()
        self.in_channel = in_channel
        self.grid_size = grid_size
        self.block_size = block_size
        self.block_gmlp_factor = block_gmlp_factor
        self.grid_gmlp_factor = grid_gmlp_factor
        self.input_proj_factor = input_proj_factor
        self.bias = bias
        self.dropout_rate = dropout_rate
        self.gridgmlpLayer = GridGmlpLayer(in_channel=self.in_channel//2, grid_size=self.grid_size, bias=self.bias,
                                           factor=self.grid_gmlp_factor, dropout_rate=self.dropout_rate)
        self.blockgmlpLayer = BlockGmlpLayer(in_channel=self.in_channel//2, block_size=self.block_size, bias=self.bias,
                                             factor=self.block_gmlp_factor, dropout_rate=self.dropout_rate)
        self.linear2=nn.Linear(self.in_channel, self.in_channel, bias=self.bias)
        self.dense = Densenet(self.in_channel, self.in_channel)

    def forward(self, x):
        shortcut = x
        u, v = x.chunk(2, axis=1)
        # GridGMLPLayer
        u = self.gridgmlpLayer(u)
        # BlockGMLPLayer
        v = self.blockgmlpLayer(v)
        x = torch.cat([u, v], dim=1)

        # Dense
        x = self.dense(x)

        x = F.dropout(x, 0.2)
        x = x + shortcut
        return x






class CSDN_Tem(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(CSDN_Tem, self).__init__()
        self.depth_conv = nn.Conv2d(
            in_channels=in_ch,
            out_channels=in_ch,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=in_ch
        )
        self.point_conv = nn.Conv2d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1
        )

    def forward(self, input):
        out = self.depth_conv(input)
        out = self.point_conv(out)
        return out

class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2

class BasicBlock(nn.Module):
    def __init__(self, intput_channel, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.2):
        super().__init__()

        dw_channel = intput_channel * DW_Expand
        self.conv1 = CSDN_Tem(intput_channel, dw_channel)
        self.conv2 = CSDN_Tem(dw_channel, dw_channel)
        self.conv3 = CSDN_Tem(dw_channel, intput_channel)


        self.PA = ResidualSplitHeadMultiAxisGmlpLayer(dw_channel)
        # self.PA = mutil_sptical_attention(dw_channel)
        # self.CA = multi_channel_attention(dw_channel)
        self.CA = multi_channel(dw_channel)

        #############################################################################
        # SimpleGate
        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * intput_channel

        self.conv4 = CSDN_Tem(intput_channel, ffn_channel)
        self.conv5 = CSDN_Tem(ffn_channel // 2, intput_channel)


        self.norm1 = LayerNorm2d(intput_channel)
        self.norm2 = LayerNorm2d(intput_channel)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, intput_channel, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, intput_channel, 1, 1)), requires_grad=True)

    def forward(self, x):
        residual = x
        x = self.norm1(x)  # channel//2
        x = self.conv1(x)
        x = self.conv2(x)
        ##################################
        x = self.CA(self.PA(x))
        ##################################
        x = self.conv3(x)
        x = self.dropout1(x)

        y = residual + x * self.beta
        x = self.norm2(y)
        x = self.conv4(x)
        x = self.sg(x)  # channel//2
        x = self.conv5(x)

        x = self.dropout2(x)
        output = y + x * self.gamma

        return output


######################################## unet
######################################
# 2  Underwater Light Field Retention : Neural Rendering for Underwater Imaging (UWNR)
#####################################
class double_conv(nn.Module):
    '''(conv => BN => ReLU) * 2'''

    def __init__(self, in_ch, out_ch, conv=nn.Conv2d, act=nn.ELU):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            conv(in_ch, out_ch, 3, padding=1),
            act(inplace=True),
            conv(out_ch, out_ch, 3, padding=1),
            act(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x) + x
        return x


class downsample(nn.Module):
    def __init__(self, in_ch, out_ch, conv=nn.Conv2d, act=nn.ELU):
        super(downsample, self).__init__()
        self.mpconv = nn.Sequential(
            conv(in_ch, out_ch, kernel_size=3, stride=2, padding=3 // 2),
            act(inplace=True),
            BasicBlock(out_ch)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x


class upsample(nn.Module):
    def __init__(self, in_ch, conv=nn.Conv2d):
        super(upsample, self).__init__()
        self.up = nn.Sequential(
            conv(in_ch, 2 * in_ch, kernel_size=3, stride=1, padding=3 // 2),
            nn.ELU(),
            nn.PixelShuffle(2),
        )

    def forward(self, x):
        y = self.up(x)
        return y

######################################################################################33


########################################################################   gamma curv
class CSDN_Tem(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(CSDN_Tem, self).__init__()
        self.depth_conv = nn.Conv2d(
            in_channels=in_ch,
            out_channels=in_ch,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=in_ch
        )
        self.point_conv = nn.Conv2d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1
        )

    def forward(self, input):
        out = self.depth_conv(input)
        out = self.point_conv(out)
        return out


class enhance_net_nopool(nn.Module):

    def __init__(self, scale_factor):
        super(enhance_net_nopool, self).__init__()

        self.relu = nn.ReLU(inplace=True)
        self.scale_factor = scale_factor
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=self.scale_factor)
        number_f = 32

        #   zerodce DWC + p-shared
        self.e_conv1 = CSDN_Tem(3, number_f)
        self.e_conv2 = CSDN_Tem(number_f, number_f)
        self.e_conv3 = CSDN_Tem(number_f, number_f)
        self.e_conv4 = CSDN_Tem(number_f, number_f)
        self.e_conv5 = CSDN_Tem(number_f * 2, number_f)
        self.e_conv6 = CSDN_Tem(number_f * 2, number_f)
        self.e_conv7 = CSDN_Tem(number_f * 2, 3)

    def enhance(self, x, x_r, x_r1):

        x = x + x_r * (torch.pow(x, 2) - x) + x_r1 * (torch.pow(x, 3) - x)
        x = x + x_r * (torch.pow(x, 2) - x) + x_r1 * (torch.pow(x, 3) - x)
        x = x + x_r * (torch.pow(x, 2) - x) + x_r1 * (torch.pow(x, 3) - x)
        enhance_image_1 = x + x_r * (torch.pow(x, 2) - x) + x_r1 * (torch.pow(x, 3) - x)

        x = enhance_image_1 + x_r * (torch.pow(enhance_image_1, 2) - enhance_image_1) + x_r1 * (
                    torch.pow(enhance_image_1, 3) - enhance_image_1)
        x = x + x_r * (torch.pow(x, 2) - x) + x_r1 * (torch.pow(x, 3) - x)
        x = x + x_r * (torch.pow(x, 2) - x) + x_r1 * (torch.pow(x, 3) - x)
        enhance_image = x + x_r * (torch.pow(x, 2) - x) + x_r1 * (torch.pow(x, 3) - x)

        return enhance_image

    def forward(self, x):
        if self.scale_factor == 1:
            x_down = x
        else:
            x_down = F.interpolate(x, scale_factor=1 / self.scale_factor, mode='bilinear')

        x1 = self.relu(self.e_conv1(x_down))
        x2 = self.relu(self.e_conv2(x1))
        x3 = self.relu(self.e_conv3(x2))
        x4 = self.relu(self.e_conv4(x3))
        x5 = self.relu(self.e_conv5(torch.cat([x3, x4], 1)))
        x6 = self.relu(self.e_conv6(torch.cat([x2, x5], 1)))
        x_r = F.tanh(self.e_conv7(torch.cat([x1, x6], 1)))

        x_r1 = F.tanh(self.e_conv7(torch.cat([x2, x5], 1)))

        if self.scale_factor == 1:
            x_r = x_r
            x_r1 = x_r1
        else:
            x_r = self.upsample(x_r)
            x_r1 = self.upsample(x_r1)
        enhance_image = self.enhance(x, x_r, x_r1)
        return enhance_image, x_r + x_r1



##########################################################################


from networks.color1 import SepLUT
from networks.feature_aggregation import FFBlock, Fusion_sample

from loss.losses import *

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.in_conv_down1 = downsample(3, 64)
        self.down2 = downsample(64, 128)
        self.down3 = downsample(128, 256)
        self.down4 = downsample(256, 512)

        self.up1 = upsample(512)
        self.up2 = upsample(256)  # in@128 out@64
        self.up3 = upsample(128)  # in@64 out@32
        self.up4 = upsample(64)  # in@16

        self.out = nn.Sequential(
            nn.Conv2d(32, 3, 3, 1, 1),
            nn.Tanh(),
        )

        # self.Color = LUTnet(64)
        self.Color = SepLUT()
        self.feature_aggregation1 = FFBlock()

        self.enhance_gama = enhance_net_nopool(1).cuda()
        self.enhance_gama.load_state_dict(torch.load('gamma/Epoch99.pth'))

        self.feature_total = Fusion_sample()

        self.texture = MultiscaleRecLoss()
        self.l1 = CharbonnierLoss()
        self.L_color = L_color()
        self.L_spa = L_spa()
        self.L_exp = L_exp(16)
        self.L_TV = L_TV()


    def forward(self, x, target, input_, gt, flat):

####################################################
##               texture
####################################################
        x2 = self.in_conv_down1(x)
        x4 = self.down2(x2)
        x8 = self.down3(x4)
        x16 = self.down4(x8)

        y = x16
        y8 = self.up1(y) + x8
        y4 = self.up2(y8) + x4
        y2 = self.up3(y4) + x2
        y = self.up4(y2)

        out_texture = self.out(y)


#########################################################
##                    gamma
######################################################


        imgs1, _ = self.enhance_gama(input_)

#########################################################
#              color
#####################################################
####### aggregation  ######################
        inp = [x4, x8, x16]
        code = self.feature_aggregation1(inp)

        out_color, loss_color = self.Color(code, imgs1, gt)


#########################################################
#             final fusion
########################################################

        out, loss_total = self.feature_total(input_, out_texture, out_color, gt)                      # 外部聚合


        if flat == 1:
            loss_texture = self.texture(out_texture, target)
            loss_color = loss_color['loss_recons'] + loss_color['loss_mono'] + loss_color['loss_sparse'] + loss_color['loss_smooth']
            loss_total = loss_total

        else:
            loss_texture = 0
            loss_color = 0
            loss_total = 0

        loss = 0.5*loss_texture + 0.5*loss_color + 10*loss_total + self.l1(out, gt)

        return out, loss





