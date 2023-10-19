import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import scipy.stats as st


def tv_loss(x, beta = 0.5, reg_coeff = 5):
    '''Calculates TV loss for an image `x`.
        
    Args:
        x: image, torch.Variable of torch.Tensor
        beta: See https://arxiv.org/abs/1412.0035 (fig. 2) to see effect of `beta` 
    '''
    dh = torch.pow(x[:,:,:,1:] - x[:,:,:,:-1], 2)
    dw = torch.pow(x[:,:,1:,:] - x[:,:,:-1,:], 2)
    a,b,c,d=x.shape
    return reg_coeff*(torch.sum(torch.pow(dh[:, :, :-1] + dw[:, :, :, :-1], beta))/(a*b*c*d))

class TVLoss(nn.Module):
    def __init__(self, tv_loss_weight=1):
        super(TVLoss, self).__init__()
        self.tv_loss_weight = tv_loss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self.tensor_size(x[:, :, 1:, :])
        count_w = self.tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return self.tv_loss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size

    @staticmethod
    def tensor_size(t):
        return t.size()[1] * t.size()[2] * t.size()[3]

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

#####################################
#
#             PerceptualLoss
#
####################################
from torchvision.models.vgg import vgg19,vgg16
class VGG19_PercepLoss(nn.Module):
    def __init__(self):
        super(VGG19_PercepLoss,self).__init__()
        self.L1 = nn.L1Loss().cuda()
        self.mse = nn.MSELoss().cuda()
        vgg = vgg19(pretrained=True).eval().cuda()
        self.loss_net1 = nn.Sequential(*list(vgg.features)[:1]).eval().cuda()
        self.loss_net3 = nn.Sequential(*list(vgg.features)[:3]).eval().cuda()
        self.loss_net5 = nn.Sequential(*list(vgg.features)[:5]).eval().cuda()
        self.loss_net9 = nn.Sequential(*list(vgg.features)[:9]).eval().cuda()
        self.loss_net13 = nn.Sequential(*list(vgg.features)[:13]).eval().cuda()
    def forward(self,x,y):
        loss1 = self.L1(self.loss_net1(x),self.loss_net1(y))
        loss3 = self.L1(self.loss_net3(x),self.loss_net3(y))
        loss5 = self.L1(self.loss_net5(x),self.loss_net5(y))
        loss9 = self.L1(self.loss_net9(x),self.loss_net9(y))
        loss13 = self.L1(self.loss_net13(x),self.loss_net13(y))
        #print(self.loss_net13(x).shape)
        loss = 0.2*loss1 + 0.2*loss3 + 0.2*loss5 + 0.2*loss9 + 0.2*loss13
        return loss
import lpips

##############################################################################
#
#                              color_Loss
#
##############################################################################
###################### angular
class ArcMarginProduct(nn.Module):
    r"""Implement of large margin arc distance: :
        Args:
            in_features: size of each input sample
            out_features: size of each output sample
            s: norm of input feature
            m: margin
            cos(theta + m)
        """

    def __init__(self, in_features=3, out_features=3, s=30.0, m=0.50, easy_margin=False):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features  # 输入特征维度
        self.out_features = out_features  # 输出特征维度
        self.s = s  # re-scale
        self.m = m  # 角度惩罚项
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))  # 权重矩阵

        nn.init.xavier_uniform_(self.weight)  # 权重矩阵初始化

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input, label):
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        # 对应伪代码中的1、2、3行：输入x标准化、输入W标准化和它们之间进行FC层得到cos(theta)
        b, c, h, w = input.shape
        label_view = label.view(b, c, h*w).permute(0, 2, 1)
        input_view = input.view(b, c, h*w).permute(0, 2, 1)
        label_norm = F.normalize(label_view, dim=-1)
        input_norm = F.normalize(input_view, dim=-1)
        cosine = torch.mul(input_norm, label_norm)
        cose_value = torch.sum(cosine, dim=-1)
        output = torch.mean(1 - cose_value)
        return output

class L_color(nn.Module):

    def __init__(self):
        super(L_color, self).__init__()

    def forward(self, x ):

        b,c,h,w = x.shape
        mean_rgb = torch.mean(x,[2,3],keepdim=True)
        mr,mg, mb = torch.split(mean_rgb, 1, dim=1)
        Drg = torch.pow(mr-mg,2)
        Drb = torch.pow(mr-mb,2)
        Dgb = torch.pow(mb-mg,2)
        k = torch.pow(torch.pow(Drg,2) + torch.pow(Drb,2) + torch.pow(Dgb,2),0.5)
        return k

class color_Loss_angular(nn.Module):
    def __init__(self):
        super(color_Loss_angular, self).__init__()
        self.angular = ArcMarginProduct()
        self.dis = nn.MSELoss()
        self.l_color = L_color()
    def forward(self, ture_reflect, pred_reflect):
        b,c,h,w = ture_reflect.shape
        # angular
        color_loss1 = self.angular(ture_reflect, pred_reflect)
        # UCIM
        color_loss2 = self.dis(ture_reflect, pred_reflect)
        # color_loss2 = torch.sum(torch.pow((pred_reflect - ture_reflect), 2)).div(2 * pred_reflect.size()[0])
        color_loss3 = torch.mean(self.l_color(pred_reflect))

        color_loss = color_loss1 + color_loss2 + color_loss3

        return color_loss

def gauss_kernel(kernlen=21, nsig=3, channels=1):
    interval = (2 * nsig + 1.) / (kernlen)
    x = np.linspace(-nsig - interval / 2., nsig + interval / 2., kernlen + 1)
    kern1d = np.diff(st.norm.cdf(x))
    kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
    kernel = kernel_raw / kernel_raw.sum()
    out_filter = np.array(kernel, dtype=np.float32)
    out_filter = out_filter.reshape((kernlen, kernlen, 1, 1))
    out_filter = np.repeat(out_filter, channels, axis=2)
    return out_filter
class Blur(nn.Module):
    def __init__(self, nc=3):
        super(Blur, self).__init__()
        self.nc = nc
        kernel = gauss_kernel(kernlen=21, nsig=3, channels=self.nc)
        kernel = torch.from_numpy(kernel).permute(2, 3, 0, 1)
        self.weight = nn.Parameter(data=kernel, requires_grad=False).cuda()

    def forward(self, x):
        if x.size(1) != self.nc:
            raise RuntimeError(
                "The channel of input [%d] does not match the preset channel [%d]" % (x.size(1), self.nc))
        x = F.conv2d(x, self.weight, stride=1, padding=10, groups=self.nc)
        return x

class color_Loss(nn.Module):
    def __init__(self):
        super(color_Loss, self).__init__()
        self.blur = Blur()
        self.color = color_Loss_angular()
    def forward(self, x1, x2):
        blur_rgb = self.blur
        blur_rgb1 = blur_rgb(x1)
        blur_rgb2 = blur_rgb(x2)

        color = self.color(blur_rgb1, blur_rgb2)
        return color



#####################

class L_color(nn.Module):

    def __init__(self):
        super(L_color, self).__init__()

    def forward(self, x):
        b, c, h, w = x.shape

        mean_rgb = torch.mean(x, [2, 3], keepdim=True)
        mr, mg, mb = torch.split(mean_rgb, 1, dim=1)
        Drg = torch.pow(mr - mg, 2)
        Drb = torch.pow(mr - mb, 2)
        Dgb = torch.pow(mb - mg, 2)
        k = torch.pow(torch.pow(Drg, 2) + torch.pow(Drb, 2) + torch.pow(Dgb, 2), 0.5)

        return k


class L_spa(nn.Module):

    def __init__(self):
        super(L_spa, self).__init__()
        # print(1)kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
        kernel_left = torch.FloatTensor([[0, 0, 0], [-1, 1, 0], [0, 0, 0]]).cuda().unsqueeze(0).unsqueeze(0)
        kernel_right = torch.FloatTensor([[0, 0, 0], [0, 1, -1], [0, 0, 0]]).cuda().unsqueeze(0).unsqueeze(0)
        kernel_up = torch.FloatTensor([[0, -1, 0], [0, 1, 0], [0, 0, 0]]).cuda().unsqueeze(0).unsqueeze(0)
        kernel_down = torch.FloatTensor([[0, 0, 0], [0, 1, 0], [0, -1, 0]]).cuda().unsqueeze(0).unsqueeze(0)
        self.weight_left = nn.Parameter(data=kernel_left, requires_grad=False)
        self.weight_right = nn.Parameter(data=kernel_right, requires_grad=False)
        self.weight_up = nn.Parameter(data=kernel_up, requires_grad=False)
        self.weight_down = nn.Parameter(data=kernel_down, requires_grad=False)
        self.pool = nn.AvgPool2d(4)

    def forward(self, org, enhance):
        b, c, h, w = org.shape

        org_mean = torch.mean(org, 1, keepdim=True)
        enhance_mean = torch.mean(enhance, 1, keepdim=True)

        org_pool = self.pool(org_mean)
        enhance_pool = self.pool(enhance_mean)

        weight_diff = torch.max(
            torch.FloatTensor([1]).cuda() + 10000 * torch.min(org_pool - torch.FloatTensor([0.3]).cuda(),
                                                              torch.FloatTensor([0]).cuda()),
            torch.FloatTensor([0.5]).cuda())
        E_1 = torch.mul(torch.sign(enhance_pool - torch.FloatTensor([0.5]).cuda()), enhance_pool - org_pool)

        D_org_letf = F.conv2d(org_pool, self.weight_left, padding=1)
        D_org_right = F.conv2d(org_pool, self.weight_right, padding=1)
        D_org_up = F.conv2d(org_pool, self.weight_up, padding=1)
        D_org_down = F.conv2d(org_pool, self.weight_down, padding=1)

        D_enhance_letf = F.conv2d(enhance_pool, self.weight_left, padding=1)
        D_enhance_right = F.conv2d(enhance_pool, self.weight_right, padding=1)
        D_enhance_up = F.conv2d(enhance_pool, self.weight_up, padding=1)
        D_enhance_down = F.conv2d(enhance_pool, self.weight_down, padding=1)

        D_left = torch.pow(D_org_letf - D_enhance_letf, 2)
        D_right = torch.pow(D_org_right - D_enhance_right, 2)
        D_up = torch.pow(D_org_up - D_enhance_up, 2)
        D_down = torch.pow(D_org_down - D_enhance_down, 2)
        E = (D_left + D_right + D_up + D_down)
        # E = 25*(D_left + D_right + D_up +D_down)

        return E


class L_exp(nn.Module):

    def __init__(self, patch_size):
        super(L_exp, self).__init__()
        # print(1)
        self.pool = nn.AvgPool2d(patch_size)
        # self.mean_val = mean_val

    def forward(self, x, mean_val):
        b, c, h, w = x.shape
        x = torch.mean(x, 1, keepdim=True)
        mean = self.pool(x)

        d = torch.mean(torch.pow(mean - torch.FloatTensor([mean_val]).cuda(), 2))
        return d


class L_TV(nn.Module):
    def __init__(self, TVLoss_weight=1):
        super(L_TV, self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = (x.size()[2] - 1) * x.size()[3]
        count_w = x.size()[2] * (x.size()[3] - 1)
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return self.TVLoss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size







##########################################################################
#
#            texture_loss
#
###################################################################

class MultiscaleRecLoss(nn.Module):
    def __init__(self, scale=3, rec_loss_type='l1', multiscale=True):
        super(MultiscaleRecLoss, self).__init__()
        self.multiscale = multiscale
        if rec_loss_type == 'l1':
            self.criterion = nn.L1Loss()
        elif rec_loss_type == 'smoothl1':
            self.criterion = nn.SmoothL1Loss()
        elif rec_loss_type == 'l2':
            self.criterion = nn.MSELoss()
        else:
            raise NotImplementedError('Loss [{}] is not implemented'.format(rec_loss_type))
        self.downsample = nn.AvgPool2d(2, stride=2, count_include_pad=False)
        if self.multiscale:
            self.weights = [1.0, 1.0/2, 1.0/4]
            self.weights = self.weights[:scale]

    def forward(self, input, target):
        loss = 0
        pred = input.clone()
        gt = target.clone()
        if self.multiscale:
            for i in range(len(self.weights)):
                loss += self.weights[i] * self.criterion(pred, gt)
                if i != len(self.weights) - 1:
                    pred = self.downsample(pred)
                    gt = self.downsample(gt)
        else:
            loss = self.criterion(pred, gt)
        return loss






