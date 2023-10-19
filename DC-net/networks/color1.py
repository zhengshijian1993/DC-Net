#!/usr/bin/env
# -*- coding: utf-8 -*-

'''
Copyright (c) 2022 Charles
'''


import torchvision
import numpy as np
import math
from loss.losses import *
from .visualize import draw_3D
from torchvision.utils import save_image


from loss.losses import color_Loss

########################################################### backbone ############


class BasicBlock(nn.Sequential):
    r"""The basic block module (Conv+LeakyReLU[+InstanceNorm]).
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, norm=False):
        body = [
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=1),
            nn.LeakyReLU(0.2)
        ]
        if norm:
            body.append(nn.InstanceNorm2d(out_channels, affine=True))
        super(BasicBlock, self).__init__(*body)


class LightBackbone(nn.Sequential):
    r"""The 5-layer CNN backbone module in [TPAMI 3D-LUT]
        (https://github.com/HuiZeng/Image-Adaptive-3DLUT).

    Args:
        input_resolution (int, optional): Resolution for pre-downsampling. Default: 256.
        extra_pooling (bool, optional): Whether to use an extra pooling layer at the end
            of the backbone. Default: False.
        n_base_feats (int, optional): Channel multiplier. Default: 8.
    """

    def __init__(self,
                 input_resolution=256,
                 extra_pooling=False,
                 n_base_feats=8,
                 **kwargs) -> None:
        body = [BasicBlock(3, n_base_feats, stride=2, norm=True)]
        n_feats = n_base_feats
        for _ in range(3):
            body.append(
                BasicBlock(n_feats, n_feats * 2, stride=2, norm=True))
            n_feats = n_feats * 2
        body.append(BasicBlock(n_feats, n_feats, stride=2))
        body.append(nn.Dropout(p=0.5))
        if extra_pooling:
            body.append(nn.AdaptiveAvgPool2d(2))
        super().__init__(*body)
        self.input_resolution = input_resolution
        self.out_channels = n_feats * (
            4 if extra_pooling else (input_resolution // 32) ** 2)

    def forward(self, imgs):
        imgs = F.interpolate(imgs, size=(self.input_resolution,) * 2,
            mode='bilinear', align_corners=False)
        return super().forward(imgs).view(imgs.shape[0], -1)


class Res18Backbone(nn.Module):
    r"""The ResNet-18 backbone.

    Args:
        pretrained (bool, optional): Whether to use the torchvison pretrained weights.
            Default: True.
        input_resolution (int, optional): Resolution for pre-downsampling. Default: 224.
    """

    def __init__(self, pretrained=True, input_resolution=224, **kwargs):
        super().__init__()
        net = torchvision.models.resnet18(pretrained=pretrained)
        net.fc = nn.Identity()
        self.net = net
        self.input_resolution = input_resolution
        self.out_channels = 512

    def forward(self, imgs):
        imgs = F.interpolate(imgs, size=(self.input_resolution,) * 2,
            mode='bilinear', align_corners=False)
        return self.net(imgs).view(imgs.shape[0], -1)


########################################  LUT  ########################3

def lut_transform(imgs, luts):
    # img (b, 3, h, w), lut (b, c, m, m, m)

    # normalize pixel values
    imgs = (imgs - .5) * 2.
    # reshape img to grid of shape (b, 1, h, w, 3)
    grids = imgs.permute(0, 2, 3, 1).unsqueeze(1)

    # after gridsampling, output is of shape (b, c, 1, h, w)
    outs = F.grid_sample(luts, grids,
        mode='bilinear', padding_mode='border', align_corners=True)
    # remove the extra dimension
    outs = outs.squeeze(2)
    return outs

class LUT3DGenerator(nn.Module):
    r"""The 3DLUT generator module.

    Args:
        n_colors (int): Number of input color channels.
        n_vertices (int): Number of sampling points along each lattice dimension.
        n_feats (int): Dimension of the input image representation vector.
        n_ranks (int): Number of ranks (or the number of basis LUTs).
    """

    def __init__(self, n_colors, n_vertices, n_feats, n_ranks) -> None:
        super().__init__()

        # h0
        self.weights_generator = nn.Linear(n_feats, n_ranks)
        # h1
        self.basis_luts_bank = nn.Linear(
            n_ranks, n_colors * (n_vertices ** n_colors), bias=False)

        self.n_colors = n_colors
        self.n_vertices = n_vertices
        self.n_feats = n_feats
        self.n_ranks = n_ranks

    def init_weights(self):
        r"""Init weights for models.

        For the mapping f (`backbone`) and h (`lut_generator`), we follow the initialization in
            [TPAMI 3D-LUT](https://github.com/HuiZeng/Image-Adaptive-3DLUT).

        """
        nn.init.ones_(self.weights_generator.bias)
        identity_lut = torch.stack([
            torch.stack(
                torch.meshgrid(*[torch.arange(self.n_vertices) for _ in range(self.n_colors)]),
                dim=0).div(self.n_vertices - 1).flip(0),
            *[torch.zeros(
                self.n_colors, *((self.n_vertices,) * self.n_colors)) for _ in range(self.n_ranks - 1)]
            ], dim=0).view(self.n_ranks, -1)
        self.basis_luts_bank.weight.data.copy_(identity_lut.t())

    def forward(self, x):
        weights = self.weights_generator(x)
        luts = self.basis_luts_bank(weights)
        luts = luts.view(x.shape[0], -1, *((self.n_vertices,) * self.n_colors))
        return weights, luts

    def regularizations(self, smoothness, monotonicity):
        basis_luts = self.basis_luts_bank.weight.t().view(
            self.n_ranks, self.n_colors, *((self.n_vertices,) * self.n_colors))
        tv, mn = 0, 0
        for i in range(2, basis_luts.ndimension()):
            diff = torch.diff(basis_luts.flip(i), dim=i)
            tv += torch.square(diff).sum(0).mean()
            mn += F.relu(diff).sum(0).mean()
        reg_smoothness = smoothness * tv
        reg_monotonicity = monotonicity * mn
        return reg_smoothness, reg_monotonicity

########################################################################

class SepLUT(nn.Module):
    r"""Separable Image-adaptive Lookup Tables for Real-time Image Enhancement.

    Args:
        n_ranks (int, optional): Number of ranks for 3D LUT (or the number of basis
            LUTs). Default: 3.
        n_vertices_3d (int, optional): Size of the 3D LUT. If `n_vertices_3d` <= 0,
            the 3D LUT will be disabled. Default: 17.
        n_vertices_1d (int, optional): Size of the 1D LUTs. If `n_vertices_1d` <= 0,
            the 1D LUTs will be disabled. Default: 17.
        lut1d_color_share (bool, optional): Whether to share a single 1D LUT across
            three color channels. Default: False.
        backbone (str, optional): Backbone architecture to use. Can be either 'light'
            or 'res18'. Default: 'light'.
        n_base_feats (int, optional): The channel multiplier of the backbone network.
            Only used when `backbone` is 'light'. Default: 8.
        pretrained (bool, optional): Whether to use ImageNet-pretrained weights.
            Only used when `backbone` is 'res18'. Default: None.
        n_colors (int, optional): Number of input color channels. Default: 3.
        sparse_factor (float, optional): Loss weight for the sparse regularization term.
            Default: 0.0001.
        smooth_factor (float, optional): Loss weight for the smoothness regularization term.
            Default: 0.
        monotonicity_factor (float, optional): Loss weight for the monotonicaity
            regularization term. Default: 10.0.
        recons_loss (dict, optional): Config for pixel-wise reconstruction loss.
        train_cfg (dict, optional): Config for training. Default: None.
        test_cfg (dict, optional): Config for testing. Default: None.
    """


    def __init__(self,
        n_ranks=3,
        n_vertices_3d=33,
        backbone='light',
        n_base_feats=8,
        pretrained=False,
        n_colors=3,
        sparse_factor=0.0001,
        smooth_factor=0.0001,
        monotonicity_factor=0.0001,

        train_cfg=None,
        test_cfg=None):

        super().__init__()

        assert backbone in ['light', 'res18']
        assert n_vertices_3d > 0

        self.backbone = dict(
            light=LightBackbone,
            res18=Res18Backbone)[backbone.lower()](
                pretrained=pretrained,
                extra_pooling=True,
                n_base_feats=n_base_feats)

        if n_vertices_3d > 0:
            self.lut3d_generator = LUT3DGenerator(
                n_colors, n_vertices_3d, self.backbone.out_channels, n_ranks)


        self.n_ranks = n_ranks
        self.n_colors = n_colors
        self.n_vertices_3d = n_vertices_3d

        self.sparse_factor = sparse_factor
        self.smooth_factor = smooth_factor
        self.monotonicity_factor = monotonicity_factor
        self.backbone_name = backbone.lower()
        self.recons_loss = color_Loss()
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.fp16_enabled = False

        self.init_weights()


    def init_weights(self):
        r"""Init weights for models.

        For the backbone network and the 3D LUT generator, we follow the initialization in
            [TPAMI 3D-LUT](https://github.com/HuiZeng/Image-Adaptive-3DLUT).
        """
        def special_initilization(m):
            classname = m.__class__.__name__
            if 'Conv' in classname:
                nn.init.xavier_normal_(m.weight.data)
            elif 'InstanceNorm' in classname:
                nn.init.normal_(m.weight.data, 1.0, 0.02)
                nn.init.constant_(m.bias.data, 0.0)
        if self.backbone_name not in ['res18']:
            self.apply(special_initilization)
        if self.n_vertices_3d > 0:
            self.lut3d_generator.init_weights()


    def forward_dummy(self, codes, imgs):
        r"""The real implementation of model forward.

        Args:
            img (Tensor): Input image, shape (b, c, h, w).
        Returns:
            tuple(Tensor, Tensor, Tensor):
                Output image, 3DLUT weights, 1DLUTs.
        """

        # context vector: (b, f)
        codes1 = self.backbone(imgs)
        codes = 0.1 * codes + codes1

        # generate 3DLUT and perform the 3D LUT transform
        if self.n_vertices_3d > 0:
            # (b, c, d, d, d)
            lut3d_weights, lut3d = self.lut3d_generator(codes)
            outs = lut_transform(imgs, lut3d)

        else:
            lut3d_weights = imgs.new_zeros(1)
            outs = imgs

        # return outs, lut3d_weights
        return outs, lut3d_weights


    def forward(self, codes, image_train, gt):
        losses = dict()
        output1, lut3d_weights = self.forward_dummy(codes, image_train)

        losses['loss_recons'] = self.recons_loss(output1, gt)

        if self.sparse_factor > 0 and lut3d_weights is not None:
            losses['loss_sparse'] = self.sparse_factor * torch.mean(lut3d_weights.pow(2))
        if self.n_vertices_3d > 0:
            reg_smoothness, reg_monotonicity = self.lut3d_generator.regularizations(
                self.smooth_factor, self.monotonicity_factor)
            if self.smooth_factor > 0:
                losses['loss_smooth'] = reg_smoothness
            if self.monotonicity_factor > 0:
                losses['loss_mono'] = reg_monotonicity

        return output1, losses




