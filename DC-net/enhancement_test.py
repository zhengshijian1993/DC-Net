from config import Config
opt = Config('training.yml')

import numpy as np
import os
import argparse
from tqdm import tqdm

import torch.nn as nn
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

import scipy.io as sio
# from networks.MIRNet_model import MIRNet                       # 设置网络
from networks.RRDBNet_arch import Generator
from dataloaders.data_rgb import get_validation_data, get_training_data
import utils
from skimage import img_as_ubyte

from thop import profile
from fvcore.nn import FlopCountAnalysis, parameter_count_table

parser = argparse.ArgumentParser(description='Image Enhancement using MIRNet')

parser.add_argument('--input_dir', default='./data/test/', type=str, help='Directory of validation images')
parser.add_argument('--result_dir', default='./checkpoints/Denoising/results/test/', type=str, help='Directory for results')
parser.add_argument('--weights', default='./checkpoints/Denoising/models/MIRNet/model_latest.pth', type=str, help='Path to weights')

parser.add_argument('--gpus', default='0', type=str, help='CUDA_VISIBLE_DEVICES')
parser.add_argument('--bs', default=1, type=int, help='Batch size for dataloader')
parser.add_argument('--save_images', default=True, help='Save Enahnced images in the result directory')


args = parser.parse_args()


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

utils.mkdir(args.result_dir)
######### DataLoaders 训练数据 ###########
img_options_train = {'patch_size':opt.TRAINING.TRAIN_PS}                            # mixup混合数据大小

# test_dataset = get_validation_data(args.input_dir)
test_dataset = get_training_data(args.input_dir, img_options_train)                      # 读取训练数据
test_loader = DataLoader(dataset=test_dataset, batch_size=args.bs, shuffle=False, num_workers=1, drop_last=True)

model_restoration = Generator().cuda()

checkpoint = torch.load(args.weights)
model_restoration.load_state_dict(checkpoint['state_dict'],False) # 修改处

# utils.load_checkpoint(model_restoration, args.weights)
# print("===>Testing using weights: ", args.weights)

model_restoration.cuda()

# model_restoration=nn.DataParallel(model_restoration)
#
# model_restoration.eval()

def pad_tensor(input):
    height_org, width_org = input.shape[2], input.shape[3]
    divide = 32

    if width_org % divide != 0 or height_org % divide != 0:

        width_res = width_org % divide
        height_res = height_org % divide
        if width_res != 0:
            width_div = divide - width_res
            pad_left = int(width_div / 2)
            pad_right = int(width_div - pad_left)
        else:
            pad_left = 0
            pad_right = 0

        if height_res != 0:
            height_div = divide - height_res
            pad_top = int(height_div / 2)
            pad_bottom = int(height_div - pad_top)
        else:
            pad_top = 0
            pad_bottom = 0

        padding = nn.ReflectionPad2d((pad_left, pad_right, pad_top, pad_bottom))
        input = padding(input).data
    else:
        pad_left = 0
        pad_right = 0
        pad_top = 0
        pad_bottom = 0

    height, width = input.shape[2], input.shape[3]
    assert width % divide == 0, 'width cant divided by stride'
    assert height % divide == 0, 'height cant divided by stride'

    return input, pad_left, pad_right, pad_top, pad_bottom


def pad_tensor_back(input, pad_left, pad_right, pad_top, pad_bottom):
    height, width = input.shape[2], input.shape[3]
    return input[:, :, pad_top: height - pad_bottom, pad_left: width - pad_right]

sumflops = 0
sumparams = 0
N = 0

with torch.no_grad():
    psnr_val_rgb = []
    for ii, data_test in enumerate(tqdm(test_loader), 0):
        rgb_gt = data_test[0].cuda()
        rgb_noisy = data_test[1].cuda()
        filenames = data_test[2]
        rgb_gt_original = data_test[4].cuda()
        rgb_noisy_original = data_test[5].cuda()

        input_noisy, pad_left_noisy, pad_right_noisy, pad_top_noisy, pad_bottom_noisy = pad_tensor(rgb_noisy)  # 这个是处理unet特征不对齐问题

        restored_ori, _ = model_restoration(input_noisy, rgb_gt, rgb_noisy_original, rgb_gt_original, flat = 0)

        restored_ori = pad_tensor_back(restored_ori, pad_left_noisy, pad_right_noisy, pad_top_noisy, pad_bottom_noisy)

        rgb_restored = torch.clamp(restored_ori, 0, 1)

        psnr_val_rgb.append(utils.batch_PSNR(rgb_restored, rgb_gt_original, 1.))

        rgb_gt = rgb_gt_original.permute(0, 2, 3, 1).cpu().detach().numpy()
        rgb_noisy = rgb_noisy_original.permute(0, 2, 3, 1).cpu().detach().numpy()
        rgb_restored = rgb_restored.permute(0, 2, 3, 1).cpu().detach().numpy()

        if args.save_images:
            for batch in range(len(rgb_gt)):
                enhanced_img = img_as_ubyte(rgb_restored[batch])
                utils.save_img(args.result_dir + filenames[batch][:-4] + '.jpg', enhanced_img)
            
psnr_val_rgb = sum(psnr_val_rgb)/len(psnr_val_rgb)
print("PSNR: %.2f " %(psnr_val_rgb))

