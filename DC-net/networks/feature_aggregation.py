import torch
from torch import nn
import numpy as np
import math
import lpips
import torch.nn.functional as F




#######################################################################
#
#
#
###########################################################################################

def weights_init(init_type='normal', gain=0.02):

    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):

            if init_type == 'normal':
                nn.init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                nn.init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                nn.init.orthogonal_(m.weight.data, gain=gain)

            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)

        elif classname.find('BatchNorm2d') != -1:
            nn.init.normal_(m.weight.data, 1.0, gain)
            nn.init.constant_(m.bias.data, 0.0)

    return init_func

class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, eps=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        # loss = torch.sum(torch.sqrt(diff * diff + self.eps))
        loss = torch.mean(torch.sqrt((diff * diff) + (self.eps*self.eps)))
        return loss

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

class GateMoudle(nn.Module):
    def __init__(self):
        super(GateMoudle, self).__init__()

        self.conv1 = nn.Conv2d(9,  64, (3, 3), 1, 1)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv2d(64, 3, (1, 1), 1, padding=0)

        for i in self.modules():
            if isinstance(i, nn.Conv2d):
                j = i.kernel_size[0] * i.kernel_size[1] * i.out_channels
                i.weight.data.normal_(0, math.sqrt(2 / j))
                if i.bias is not None:
                    i.bias.data.zero_()

        self.sg = nn.Sigmoid()
    def forward(self, x):
        con1 = self.relu(self.conv1(x))
        scoremap = self.sg(self.conv2(con1))
        return scoremap

class Fusion_sample(nn.Module):

    def __init__(self,  init_weights=True):
        super(Fusion_sample, self).__init__()

        self.tv_loss = CharbonnierLoss()
        # self.content_loss = lpips.LPIPS(net="vgg")
        self.content_loss = nn.MSELoss()

        self.covn_ = CSDN_Tem(3, 64)

        self.out_layer = nn.Sequential(
            CSDN_Tem(64 + 64, 64),
            nn.LeakyReLU(negative_slope=0.2),
            CSDN_Tem(64, 3),
            nn.Tanh()
        )


        self.covn4 =  CSDN_Tem(3, 64)
        self.covn5 =  CSDN_Tem(64, 3)

        self.sg = GateMoudle()                       # GateMoudle()
        self.covn6 = CSDN_Tem(9, 3)

        if init_weights:
            self.apply(weights_init())



    def forward(self,image_input, input_image1, input_image2, gt):
        h, w = input_image2.shape[2], input_image2.shape[3]
        input_image1 = F.interpolate(input_image1, (h, w), mode="nearest")


        scoremap = self.sg(torch.cat((image_input, input_image1, input_image2),1))
        fature1 = torch.mul(scoremap, input_image1)
        fature2 = torch.mul(scoremap, input_image2)
        output = torch.add(fature1, fature2)


        losses_toal = 1 * self.tv_loss(output, gt) + 0.5 * self.content_loss(output, gt)

        return output, losses_toal





########################################################################################
##
##
##
########################################################################################

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

class downsample(nn.Module):
    def __init__(self, in_ch, out_ch, conv=nn.Conv2d, act=nn.ELU):
        super(downsample, self).__init__()
        self.mpconv = nn.Sequential(
            conv(in_ch, out_ch, kernel_size=3, stride=2, padding=3 // 2),
            act(inplace=True),
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x

class upsample(nn.Module):
    def __init__(self, in_ch, conv=nn.Conv2d):
        super(upsample, self).__init__()
        self.up = nn.Sequential(
            conv(in_ch, 4 * in_ch, kernel_size=3, stride=1, padding=3 // 2),
            nn.ELU(),
            nn.PixelShuffle(2),
        )

    def forward(self, x):
        y = self.up(x)
        return y



class FFBlock(nn.Module):
    def __init__(self):
        super(FFBlock, self).__init__()


        self.body1 =  BasicBlock(128, 256, stride=1, norm=True)
        self.body2 = BasicBlock(256, 256, stride=1, norm=True)
        self.body3 = BasicBlock(512, 256, stride=1, norm=True)

        self.down = downsample(256,256)
        self.up = upsample(256)


        self.drop = nn.Dropout(p=0.5)
        self.pool = nn.AdaptiveAvgPool2d(1)
        # Init weights
        self.epsilon = 0.001  # 防止融合权重归一化时除0
        self.w1 = nn.Parameter(torch.ones((2, 2), dtype=torch.float32))
        self.w2 = nn.Parameter(torch.ones((3, 1), dtype=torch.float32))

    def forward(self, inputs):
         p_2, p_3, p_4 = inputs


         w1 = F.relu(self.w1)                                       # 保为非负   weight
         w1 /= torch.sum(w1, dim=0) + self.epsilon
         w2 = F.relu(self.w2)
         w2 /= torch.sum(w2, dim=0) + self.epsilon


         f2 = self.body1(p_2)
         f3 = self.body2(p_3)
         f4 = self.body3(p_4)



         p1 = self.body2(self.up(f4) * w1[0, 0] + f3 * w1[1, 0])
         p2 = f3
         p3 = self.body2(self.down(f2) * w1[0, 1] + f3 * w1[1, 1])

         out = self.body2(p1 * w2[0, 0] + p2 * w2[1, 0] + p3 * w2[2, 0])



         out = self.pool(out).view(out.shape[0], -1)

         return out




