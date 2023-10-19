# -*- coding: utf-8 -*-
import einops
import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
import math


#########################################################################
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

class SimpleGate(nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        # self.conv1 = nn.Conv2d(in_channels=num_channels, out_channels=num_channels*2, kernel_size=1, padding=0, stride=1,bias=True)
        self.conv1 = CSDN_Tem(num_channels, num_channels*2)
    def forward(self, x):
        x = self.conv1(x)
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2

class GetSpatialGatingWeights(nn.Module):  # n, h, w, c
    """Get gating weights for cross-gating MLP block."""

    def __init__(self, num_channels, grid_size, block_size, input_proj_factor=2, use_bias=True, dropout_rate=0):
        super().__init__()
        self.num_channels = num_channels
        self.grid_size = grid_size
        self.block_size = block_size
        self.gh = self.grid_size[0]
        self.gw = self.grid_size[1]
        self.fh = self.block_size[0]
        self.fw = self.block_size[1]
        self.input_proj_factor = input_proj_factor
        self.use_bias = use_bias
        self.drop = dropout_rate
        self.LayerNorm_in = Layer_norm_process(self.num_channels)
        self.in_project = nn.Linear(self.num_channels, self.num_channels * self.input_proj_factor, bias=self.use_bias)
        self.gelu = nn.GELU()
        self.Dense_0 = nn.Linear(self.gh * self.gw, self.gh * self.gw, bias=self.use_bias)
        self.Dense_1 = nn.Linear(self.fh * self.fw, self.fh * self.fw, bias=self.use_bias)
        self.out_project = nn.Linear(self.num_channels * self.input_proj_factor, self.num_channels, bias=self.use_bias)
        self.dropout = nn.Dropout(self.drop)

    def forward(self, x):
        _, h, w, _ = x.shape
        # input projection
        x = self.LayerNorm_in(x)
        x = self.in_project(x)  # channel projection
        x = self.gelu(x)
        c = x.size(-1) // 2
        u, v = torch.split(x, c, dim=-1)
        # get grid MLP weights
        fh, fw = h // self.gh, w // self.gw
        u = block_images_einops(u, patch_size=(fh, fw))  # n, (gh gw) (fh fw) c
        u = u.permute(0, 3, 2, 1)  # n, c, (fh fw) (gh gw)
        u = self.Dense_0(u)
        u = u.permute(0, 3, 2, 1)  # n, (gh gw) (fh fw) c
        u = unblock_images_einops(u, grid_size=(self.gh, self.gw), patch_size=(fh, fw))
        # get block MLP weights
        gh, gw = h // self.fh, w // self.fw
        v = block_images_einops(v, patch_size=(self.fh, self.fw))  # n, (gh gw) (fh fw) c
        v = v.permute(0, 1, 3, 2)  # n (gh gw) c (fh fw)
        v = self.Dense_1(v)
        v = v.permute(0, 1, 3, 2)  # n, (gh gw) (fh fw) c
        v = unblock_images_einops(v, grid_size=(gh, gw), patch_size=(self.fh, self.fw))

        x = torch.cat([u, v], dim=-1)
        x = self.out_project(x)
        x = self.dropout(x)
        return x

class CrossGatingBlock(nn.Module):  #input shape: n, c, h, w
    """Cross-gating MLP block."""
    def __init__(self, x_features, num_channels, block_size, grid_size, cin_y=0,upsample_y=True, use_bias=True, use_global_mlp=True, dropout_rate=0):
        super().__init__()
        self.cin_y = cin_y
        self.x_features = x_features
        self.num_channels = num_channels
        self.block_size = block_size
        self.grid_size = grid_size
        self.upsample_y = upsample_y
        self.use_bias = use_bias
        self.use_global_mlp = use_global_mlp
        self.drop = dropout_rate
        self.ConvTranspose_0 = nn.ConvTranspose2d(self.num_channels,self.num_channels,kernel_size=(2,2),stride=2,bias=self.use_bias)
        self.Conv_0 = nn.Conv2d(self.x_features, self.num_channels, kernel_size=(1,1),stride=1, bias=self.use_bias)
        self.Conv_1 = nn.Conv2d(self.num_channels, self.num_channels, kernel_size=(1,1),stride=1, bias=self.use_bias)
        self.LayerNorm_x = Layer_norm_process(self.num_channels)
        self.in_project_x = nn.Linear(self.num_channels, self.num_channels, bias=self.use_bias)
        self.gelu1 = nn.GELU()
        self.SplitHeadMultiAxisGating_x = GetSpatialGatingWeights(num_channels=self.num_channels,block_size=self.block_size,grid_size=self.grid_size,
            dropout_rate=self.drop,use_bias=self.use_bias)
        self.LayerNorm_y = Layer_norm_process(self.num_channels)
        self.in_project_y = nn.Linear(self.num_channels, self.num_channels, bias=self.use_bias)
        self.gelu2 = nn.GELU()
        self.SplitHeadMultiAxisGating_y = GetSpatialGatingWeights(num_channels=self.num_channels,block_size=self.block_size,grid_size=self.grid_size,
            dropout_rate=self.drop,use_bias=self.use_bias)
        self.out_project_y = nn.Linear(self.num_channels, self.num_channels, bias=self.use_bias)
        self.dropout1 = nn.Dropout(self.drop)
        self.out_project_x = nn.Linear(self.num_channels, self.num_channels, bias=self.use_bias)
        self.dropout2 = nn.Dropout(self.drop)
    def forward(self, x,y):
        # Upscale Y signal, y is the gating signal.
        if self.upsample_y:
                y = self.ConvTranspose_0(y)
        x = self.Conv_0(x)
        y = self.Conv_1(y)
        assert y.shape == x.shape
        x = x.permute(0,2,3,1)  #n,h,w,c
        y = y.permute(0,2,3,1)  #n,h,w,c
        shortcut_x = x
        shortcut_y = y
        # Get gating weights from X
        x = self.LayerNorm_x(x)
        x = self.in_project_x(x)
        x = self.gelu1(x)
        gx = self.SplitHeadMultiAxisGating_x(x)
        # Get gating weights from Y
        y = self.LayerNorm_y(y)
        y = self.in_project_y(y)
        y = self.gelu2(y)
        gy = self.SplitHeadMultiAxisGating_y(y)
        # Apply cross gating
        y = y * gx  ## gating y using x
        y = self.out_project_y(y)
        y = self.dropout1(y)
        y = y + shortcut_y
        x = x * gy  # gating x using y
        x = self.out_project_x(x)
        x = self.dropout2(x)
        x = x + y + shortcut_x  # get all aggregated signals
        return x.permute(0,3,1,2), y.permute(0,3,1,2)  #n,c,h,w


class Layer_norm_process(nn.Module):  #n, h, w, c
    def __init__(self, c, eps=1e-6):
        super().__init__()
        self.beta = torch.nn.Parameter(torch.zeros(c), requires_grad=True)
        self.gamma = torch.nn.Parameter(torch.ones(c), requires_grad=True)
        self.eps = eps
    def forward(self, feature):
        var_mean = torch.var_mean(feature, dim=-1, unbiased=False)
        mean = var_mean[1]
        var = var_mean[0]
        # layer norm process
        feature = (feature - mean[..., None]) / torch.sqrt(var[..., None] + self.eps)
        gamma = self.gamma.expand_as(feature)
        beta = self.beta.expand_as(feature)
        feature = feature * gamma + beta
        return feature

def block_images_einops(x, patch_size):  #n, h, w, c
  """Image to patches."""
  batch, height, width, channels = x.shape
  grid_height = height // patch_size[0]
  grid_width = width // patch_size[1]
  x = einops.rearrange(
      x, "n (gh fh) (gw fw) c -> n (gh gw) (fh fw) c",
      gh=grid_height, gw=grid_width, fh=patch_size[0], fw=patch_size[1])
  return x


def unblock_images_einops(x, grid_size, patch_size):
  """patches to images."""
  x = einops.rearrange(
      x, "n (gh gw) (fh fw) c -> n (gh fh) (gw fw) c",
      gh=grid_size[0], gw=grid_size[1], fh=patch_size[0], fw=patch_size[1])
  return x


class BlockGatingUnit(nn.Module):  # input shape: n (gh gw) (fh fw) c
    """A SpatialGatingUnit as defined in the gMLP paper.
    The 'spatial' dim is defined as the second last.
    If applied on other dims, you should swapaxes first.
    """

    def __init__(self, c, n, use_bias=True):
        super().__init__()
        self.c = c
        self.n = n
        self.use_bias = use_bias
        self.Dense_0 = nn.Linear(self.n, self.n, self.use_bias)
        self.intermediate_layernorm = Layer_norm_process(self.c // 2)

    def forward(self, x):
        c = x.size(-1)
        c = c // 2  # split size
        u, v = torch.split(x, c, dim=-1)
        v = self.intermediate_layernorm(v)
        v = v.permute(0, 1, 3, 2)  # n, (gh gw), c/2, (fh fw)
        v = self.Dense_0(v)  # apply fc on the last dimension (fh fw)
        v = v.permute(0, 1, 3, 2)  # n (gh gw) (fh fw) c/2
        return u * (v + 1.)

class GridGatingUnit(nn.Module):  # input shape: n (gh gw) (fh fw) c
    """A SpatialGatingUnit as defined in the gMLP paper.
    The 'spatial' dim is defined as the second.
    If applied on other dims, you should swapaxes first.
    """

    def __init__(self, c, n, use_bias=True):
        super().__init__()
        self.c = c
        self.n = n
        self.use_bias = use_bias
        self.intermediate_layernorm = Layer_norm_process(self.c // 2)
        self.Dense_0 = nn.Linear(self.n, self.n, self.use_bias)

    def forward(self, x):
        c = x.size(-1)
        c = c // 2  # split size
        u, v = torch.split(x, c, dim=-1)
        v = self.intermediate_layernorm(v)
        v = v.permute(0, 3, 2, 1)  # n, c/2, (fh fw) (gh gw)
        v = self.Dense_0(v)  # apply fc on the last dimension (gh gw)
        v = v.permute(0, 3, 2, 1)  # n (gh gw) (fh fw) c/2
        return u * (v + 1.)

class GridGmlpLayer(nn.Module):  # input shape: n, h, w, c
    """Grid gMLP layer that performs global mixing of tokens."""

    def __init__(self, grid_size, num_channels, use_bias=True, factor=2, dropout_rate=0):
        super().__init__()
        self.grid_size = grid_size
        self.gh = grid_size[0]
        self.gw = grid_size[1]
        self.num_channels = num_channels
        self.use_bias = use_bias
        self.factor = factor
        self.dropout_rate = dropout_rate
        self.LayerNorm = Layer_norm_process(self.num_channels)
        self.in_project = nn.Linear(self.num_channels, self.num_channels * self.factor, self.use_bias)  # c->c*factor
        self.gelu = nn.GELU()
        self.GridGatingUnit = GridGatingUnit(self.num_channels * self.factor,
                                             n=self.gh * self.gw)  # number of channels????????????????
        self.out_project = nn.Linear(self.num_channels * self.factor // 2, self.num_channels,
                                     self.use_bias)  # c*factor->c
        self.dropout = nn.Dropout(self.dropout_rate)

    def forward(self, x):
        n, h, w, num_channels = x.shape
        fh, fw = h // self.gh, w // self.gw
        x = block_images_einops(x, patch_size=(fh, fw))  # n (gh gw) (fh fw) c
        # gMLP1: Global (grid) mixing part, provides global grid communication.
        y = self.LayerNorm(x)
        y = self.in_project(y)  # channel proj
        y = self.gelu(y)
        y = self.GridGatingUnit(y)
        y = self.out_project(y)
        y = self.dropout(y)
        x = x + y
        x = unblock_images_einops(x, grid_size=(self.gh, self.gw), patch_size=(fh, fw))
        return x

class BlockGmlpLayer(nn.Module):  # input shape: n, h, w, c
    """Block gMLP layer that performs local mixing of tokens."""

    def __init__(self, block_size, num_channels, use_bias=True, factor=2, dropout_rate=0):
        super().__init__()
        self.block_size = block_size
        self.fh = self.block_size[0]
        self.fw = self.block_size[1]
        self.num_channels = num_channels
        self.use_bias = use_bias
        self.factor = factor
        self.dropout_rate = dropout_rate
        self.LayerNorm = Layer_norm_process(self.num_channels)
        self.in_project = nn.Linear(self.num_channels, self.num_channels * self.factor, self.use_bias)  # c->c*factor
        self.gelu = nn.GELU()
        self.BlockGatingUnit = BlockGatingUnit(self.num_channels * self.factor,
                                               n=self.fh * self.fw)  # number of channels????????????????
        self.out_project = nn.Linear(self.num_channels * self.factor // 2, self.num_channels,
                                     self.use_bias)  # c*factor->c
        self.dropout = nn.Dropout(self.dropout_rate)

    def forward(self, x):
        _, h, w, _ = x.shape
        gh, gw = h // self.fh, w // self.fw
        x = block_images_einops(x, patch_size=(self.fh, self.fw))  # n (gh gw) (fh fw) c
        # gMLP2: Local (block) mixing part, provides local block communication.
        y = self.LayerNorm(x)
        y = self.in_project(y)  # channel proj
        y = self.gelu(y)
        y = self.BlockGatingUnit(y)
        y = self.out_project(y)
        y = self.dropout(y)
        x = x + y
        x = unblock_images_einops(x, grid_size=(gh, gw), patch_size=(self.fh, self.fw))
        return x


class mutil_sptical_attention(nn.Module):   #input shape: n, h, w, c
    def __init__(self, num_channels, block_size=[2,2], grid_size=[2,2], input_proj_factor=2,block_gmlp_factor=2,grid_gmlp_factor=2,use_bias=True,dropout_rate=0.):
        super().__init__()
        self.block_size = block_size
        self.grid_size = grid_size
        self.num_channels = num_channels
        self.input_proj_factor = input_proj_factor
        self.block_gmlp_factor = block_gmlp_factor
        self.grid_gmlp_factor = grid_gmlp_factor
        self.use_bias = use_bias
        self.drop = dropout_rate
        self.LayerNorm_in = Layer_norm_process(self.num_channels)
        self.in_project = nn.Linear(self.num_channels, self.num_channels * self.input_proj_factor, bias=self.use_bias)
        self.gelu = nn.GELU()
        self.GridGmlpLayer = GridGmlpLayer(grid_size=self.grid_size,
                                           num_channels=self.num_channels * self.input_proj_factor // 2,
                                           use_bias=self.use_bias, factor=self.grid_gmlp_factor)
        self.BlockGmlpLayer = BlockGmlpLayer(block_size=self.block_size,
                                             num_channels=self.num_channels * self.input_proj_factor // 2,
                                             use_bias=self.use_bias, factor=self.block_gmlp_factor)
        self.out_project = nn.Linear(self.num_channels * self.input_proj_factor, self.num_channels, bias=self.use_bias)
        self.dropout = nn.Dropout(self.drop)

        self.CGB = CrossGatingBlock(x_features=self.num_channels, num_channels=self.num_channels, block_size=self.block_size,
                            grid_size=self.grid_size, upsample_y=False, dropout_rate=self.drop, use_bias=self.use_bias, use_global_mlp=True)

    def forward(self, x):
        shortcut = x

        x = x.permute(0,2,3,1)  # n,h,w,c

        x = self.LayerNorm_in(x)
        x = self.in_project(x)
        x = self.gelu(x)
        c = x.size(-1) // 2
        u, v = torch.split(x, c, dim=-1)
        # grid gMLP
        u = self.GridGmlpLayer(u)
        # block gMLP
        v = self.BlockGmlpLayer(v)
        # out projection

        u, v = self.CGB(u.permute(0, 3, 1, 2), v.permute(0, 3, 1, 2))
        x = torch.cat([u.permute(0,2,3,1), v.permute(0,2,3,1)], dim=-1)


        x = self.out_project(x)
        x = self.dropout(x)

        x = x.permute(0, 3, 1, 2)  #  n , c, h, w
        x = x + shortcut
        return x


#########################################################################################################
class SimplifiedChannelModule(nn.Module):
    def __init__(self, channels):
        super(SimplifiedChannelModule, self).__init__()
        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )
    def forward(self, x):
        out = self.sca(x)
        return out*x


class ChannelAttention(nn.Module):
    def __init__(self, chns, factor):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(chns, chns//factor, kernel_size=1, padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(chns//factor, chns, kernel_size=1, padding=0)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        out = self.avg_pool(x)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        weight = self.sigmoid(out)
        return weight

def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1, groups=1):
    """standard convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                     padding=padding, dilation=dilation, groups=groups, bias=False)


class bolckModule(nn.Module):

    def __init__(self, inplans, planes):
        super(bolckModule, self).__init__()
        self.conv_t = conv(inplans, planes, kernel_size=3, padding=3//2, stride=1)
        self.se = SimplifiedChannelModule(planes // 4)
        #########################################通道注意力se
        self.se_blocks = ChannelAttention(planes // 4, 4)
        self.softmax = nn.Softmax(dim=1)
        self.split_channel = planes // 4

    def forward(self, x):
        #step1:channel split
        batch_size = x.shape[0]
        feats = self.conv_t(x)

        # step2:channel--block channel
        feats = feats.view(batch_size, 4, self.split_channel, feats.shape[2], feats.shape[3])
        x1_1 = feats[:, 0, :, :, :]
        x2_2 = feats[:, 1, :, :, :]
        x3_3 = feats[:, 2, :, :, :]
        x4_4 = feats[:, 3, :, :, :]
        # step3: SE weight
        x1_se = self.se_blocks(x1_1)
        x2_se = self.se_blocks(x2_2)
        x3_se = self.se_blocks(x3_3)
        x4_se = self.se_blocks(x4_4)
        x_se = torch.cat((x1_se, x2_se, x3_se, x4_se), dim=1)

        # Step4:Softmax
        attention_vectors = x_se.view(batch_size, 4, self.split_channel, 1, 1)
        attention_vectors = self.softmax(attention_vectors)
        feats_weight = feats * attention_vectors
        for i in range(4):
            x_se_weight_fp = feats_weight[:, i, :, :]
            if i == 0:
                out = x_se_weight_fp
            else:
                out = torch.cat((x_se_weight_fp, out), 1)
        return out

class gridModule(nn.Module):                                            # PSAnet，PAnet--很像

    def __init__(self, inplans, planes):
        super(gridModule, self).__init__()

        self.se = SimplifiedChannelModule(planes // 4)
        self.conv_t = conv(inplans, planes, kernel_size=3, padding=3 // 2, stride=1)

#########################################通道注意力se
        self.se_blocks = ChannelAttention(planes // 4, 4)
        self.split_channel = planes // 4
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        #step1:channel split
        batch_size = x.shape[0]
        feats = self.conv_t(x)
        # step2:channel--grid channel
        B, C, H, W = feats.size()
        channel_per_group = C // 4
        # group 通道分组
        feats = feats.view(B, 4, channel_per_group, H, W)
        # channel shuffle 通道洗牌
        feats = torch.transpose(feats, 1, 2).contiguous()
        feats = feats.view(B, -1, H, W)

        feats = feats.view(B, 4, channel_per_group, H, W)
        x1_1 = feats[:, 0, :, :, :]
        x2_2 = feats[:, 1, :, :, :]
        x3_3 = feats[:, 2, :, :, :]
        x4_4 = feats[:, 3, :, :, :]

        # step3: SE weight
        x1_se = self.se_blocks(x1_1)
        x2_se = self.se_blocks(x2_2)
        x3_se = self.se_blocks(x3_3)
        x4_se = self.se_blocks(x4_4)
        x_se = torch.cat((x1_se, x2_se, x3_se, x4_se), dim=1)

        # Step4:Softmax
        attention_vectors = x_se.view(batch_size, 4, self.split_channel, 1, 1)
        attention_vectors = self.softmax(attention_vectors)
        feats_weight = feats * attention_vectors
        for i in range(4):
            x_se_weight_fp = feats_weight[:, i, :, :]
            if i == 0:
                out = x_se_weight_fp
            else:
                out = torch.cat((x_se_weight_fp, out), 1)
        return out



class multi_channel_attention(nn.Module):
    def __init__(self, num_channels, use_bias = True, dropout_rate = 0.):
        super().__init__()
        self.num_channels = num_channels
        self.use_bias = use_bias
        self.drop = dropout_rate

        self.LayerNorm_in = Layer_norm_process(self.num_channels)
        # self.in_project = nn.Linear(self.num_channels, self.num_channels, bias=self.use_bias)
        self.in_project = SimpleGate(self.num_channels)
        self.gelu = nn.GELU()
        self.gridLayer = gridModule(self.num_channels, self.num_channels)
        self.blocLayer = bolckModule(self.num_channels, self.num_channels)

        self.out_project = SimpleGate(self.num_channels)
        self.dropout = nn.Dropout(self.drop)

        self.CGB = CrossGatingBlock(x_features=self.num_channels, num_channels=self.num_channels,
                                block_size=[2,2],
                                grid_size=[2,2], upsample_y=False, dropout_rate=0.,
                                use_bias=True, use_global_mlp=True)

    def forward(self, x):
        shortcut = x
        # x = self.LayerNorm_in(x)
        x = self.in_project(x)
        x = self.gelu(x)
        u, v = x.chunk(2, axis=3)

        u = self.gridLayer(u)
        v =  self.blocLayer(v)
        # out projection

        # u, v = self.gat_black(u, v)
        u, v = self.CGB(u, v)

        x = torch.cat([u, v], dim=3)


        x = self.out_project(x)
        x = self.dropout(x)
        out = x + shortcut


        return out

#######################################################################################################

class bolckModule1(nn.Module):

    def __init__(self, c, b=1, gamma=2):
        super(bolckModule1, self).__init__()
        t = int(abs((math.log(c, 2) + b) / gamma))
        k = t if t % 2 else t + 1

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv1d(1, 1, kernel_size=k, padding=int(k / 2), bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        input = x
        x = self.avg_pool(x)

        x = self.conv1(x.permute(1, 2, 0))
        x = x.permute(2, 0, 1)

        out = self.sigmoid(x)
        return out * input


class gridModule1(nn.Module):  # PSAnet，PAnet--很像

    def __init__(self, c, b=1, gamma=2):
        super(gridModule1, self).__init__()
        t = int(abs((math.log(c, 2) + b) / gamma))
        k = t if t % 2 else t + 1

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv1d(1, 1, kernel_size=k, padding=int(k / 2), bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        input = x
        x = self.avg_pool(x)
        ############################ channel shuffle ##############################3
        matrix = x.cpu().detach().numpy()
        matrix = np.random.permutation(matrix)
        x = torch.from_numpy(matrix).cuda()

        ################################################################################
        x = self.conv1(x.permute(1, 2, 0))
        x = x.permute(2, 0, 1)
        out = self.sigmoid(x)
        return out * input


def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1, groups=1):
    """standard convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                     padding=padding, dilation=dilation, groups=groups, bias=False)


class SimpleGate1(nn.Module):
    def __init__(self, num_channels):
        super().__init__()

    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class multi_channel(nn.Module):
    def __init__(self, num_channels, use_bias=True, dropout_rate=0.):
        super(multi_channel, self).__init__()

        self.num_channels = num_channels
        self.use_bias = use_bias
        self.drop = dropout_rate
        self.in_project = SimpleGate(self.num_channels)
        self.gelu = nn.GELU()

        self.gridLayer = gridModule1(self.num_channels)
        self.blocLayer = bolckModule1(self.num_channels)

        self.out_project = SimpleGate(self.num_channels)
        self.out_project1 = SimpleGate1(self.num_channels)

        self.dropout = nn.Dropout(self.drop)

        self.dense = Densenet(self.num_channels * 2, self.num_channels)

    def forward(self, input):
        shortcut = input

        N, _, H, W = input.shape

        input = self.in_project(input)
        input = self.gelu(input)

        ################ mask #######################  spilt #######################33
        matrix1 = torch.zeros(int(H / 2 * W))
        matrix2 = torch.ones(int(H / 2 * W))
        matrix = torch.cat((matrix1, matrix2), dim=0)
        matrix = matrix.numpy()
        matrix = np.random.permutation(matrix)
        matrix = torch.from_numpy(matrix)
        matrix1 = matrix.view(H, W).cuda()
        matrix2 = 1 - matrix1
        #############################################
        input1 = (matrix1.unsqueeze(0) * input.squeeze(0))
        input2 = (matrix2.unsqueeze(0) * input.squeeze(0))

        u = self.gridLayer(input1).unsqueeze(0)

        v = self.blocLayer(input2).unsqueeze(0)

        out = torch.cat([u, v], dim=1)
        # out = u + v

        out = self.dense(out)
        out = F.dropout(out, 0.2)
        out = out + shortcut

        return out







