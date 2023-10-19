import os
from config import Config 
opt = Config('training.yml')

gpus = ','.join([str(i) for i in opt.GPU])
os.environ["CUDA_VISIBLE_DEVICES"] = gpus
import torch
torch.backends.cudnn.benchmark = True

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import random

from utils.util import *
import utils
from dataloaders.data_rgb import get_training_data, get_validation_data


from networks.RRDBNet_arch import Generator
from tqdm import tqdm 
from warmup_scheduler import GradualWarmupScheduler
import time


######### Set Seeds ###########
random.seed(1234)
np.random.seed(1234)
torch.manual_seed(1234)
torch.cuda.manual_seed_all(1234)

start_epoch = 1
mode = opt.MODEL.MODE                 # 模型名称
session = opt.MODEL.SESSION           # 模型模块

result_dir = os.path.join(opt.TRAINING.SAVE_DIR, mode, 'results', session)         # 结果保存路径
model_dir  = os.path.join(opt.TRAINING.SAVE_DIR, mode, 'models',  session)         # 模型保存路径

utils.mkdir(result_dir)
utils.mkdir(model_dir)

train_dir = opt.TRAINING.TRAIN_DIR            # 训练图像路径
val_dir   = opt.TRAINING.VAL_DIR              # 验证图像路径
save_images = opt.TRAINING.SAVE_IMAGES        # 保存图像路径

######### Model ###########
model_restoration = Generator().cuda()

new_lr = opt.OPTIM.LR_INITIAL

optimizer = optim.Adam(model_restoration.parameters(), lr=new_lr, betas=(0.9, 0.999),eps=1e-8, weight_decay=1e-8)

######### Scheduler 动态调整学习率 ###########
if opt.OPTIM.WARMUP:
    warmup_epochs = 3
    scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, opt.OPTIM.NUM_EPOCHS-warmup_epochs, eta_min=1e-6)   # 余弦退火调整学习率
    scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=warmup_epochs, after_scheduler=scheduler_cosine)
    scheduler.step()

######### Resume 继续训练 ###########
if opt.TRAINING.RESUME:
    path_chk_rest = utils.get_last_path(model_dir, '_latest.pth')
    utils.load_checkpoint(model_restoration,path_chk_rest)
    start_epoch = utils.load_start_epoch(path_chk_rest) + 1
    utils.load_optim(optimizer, path_chk_rest)

    for i in range(1, start_epoch):
        scheduler.step()
    new_lr = scheduler.get_lr()[0]
    print('------------------------------------------------------------------------------')
    print("==> Resuming Training with learning rate:", new_lr)
    print('------------------------------------------------------------------------------')


# criterion = generator_loss_func().cuda()
######### DataLoaders 训练数据 ###########
img_options_train = {'patch_size':opt.TRAINING.TRAIN_PS}                            # mixup混合数据大小

train_dataset = get_training_data(train_dir, img_options_train)                      # 读取训练数据
train_loader = DataLoader(dataset=train_dataset, batch_size=opt.OPTIM.BATCH_SIZE, shuffle=True, num_workers=4, drop_last=True)

val_dataset = get_training_data(val_dir, img_options_train)                                           # 验证数据集
val_loader = DataLoader(dataset=val_dataset, batch_size=1, shuffle=False, num_workers=4, drop_last=True)

print('===> Start Epoch {} End Epoch {}'.format(start_epoch,opt.OPTIM.NUM_EPOCHS + 1))
print('===> Loading datasets')

mixup = utils.MixUp_AUG()
best_psnr = 0
best_epoch = 0
best_iter = 0

eval_now = len(train_loader)//4 - 1
print(f"\nEvaluation after every {eval_now} Iterations !!!\n")

###################################### train_loop ################################
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


for epoch in range(start_epoch, opt.OPTIM.NUM_EPOCHS + 1):
    epoch_start_time = time.time()
    epoch_loss = 0
    train_id = 1

    for i, data in enumerate(tqdm(train_loader), 0):

        # zero_grad
        for param in model_restoration.parameters():
            param.grad = None

        target = data[0].cuda()
        input_ = data[1].cuda()

        target_original = data[4].cuda()
        input_original = data[5].cuda()


        if epoch>5:
            target, input_ = mixup.aug(target, input_)
#######################################################33


        restored_ori, loss = model_restoration(input_, target, input_original, target_original, flat=1)


        loss.backward()
        optimizer.step()
        epoch_loss +=loss.item()

        #### Evaluation ####
        if i%eval_now==0 and i>0:
            model_restoration.eval()
            with torch.no_grad():
                psnr_val_rgb = []
                for ii, data_val in enumerate((val_loader), 0):
                    target = data_val[0].cuda()
                    input_ = data_val[1].cuda()
                    filenames = data_val[2]
                    target_original = data_val[4].cuda()
                    input_original = data_val[5].cuda()


                    input_noisy, pad_left_noisy, pad_right_noisy, pad_top_noisy, pad_bottom_noisy = pad_tensor(input_)     # 这个是处理unet特征不对齐问题

                    restored_ori, _ = model_restoration(input_noisy, target, input_original, target_original, flat = 0)

                    restored_ori = pad_tensor_back(restored_ori, pad_left_noisy, pad_right_noisy, pad_top_noisy, pad_bottom_noisy)
                    restored = torch.clamp(restored_ori,0,1).cuda()

                    psnr_val_rgb.append(utils.batch_PSNR(restored, target_original, 1.))

                    if save_images:
                        target = target_original.permute(0, 2, 3, 1).cpu().detach().numpy()
                        input_ = input_original.permute(0, 2, 3, 1).cpu().detach().numpy()
                        restored = restored.permute(0, 2, 3, 1).cpu().detach().numpy()

                        for batch in range(input_.shape[0]):
                            temp = np.concatenate((input_[batch]*255, restored[batch]*255, target[batch]*255), axis=1)
                            # temp =  restored[batch]*255
                            utils.save_img(os.path.join(result_dir, filenames[batch][:-4] + '.jpg'), temp.astype(np.uint8))
                psnr_val_rgb = sum(psnr_val_rgb)/len(psnr_val_rgb)

                if psnr_val_rgb > best_psnr:
                    best_psnr = psnr_val_rgb
                    best_epoch = epoch
                    best_iter = i
                    torch.save({'epoch': epoch,
                                'state_dict': model_restoration.state_dict(),
                                'optimizer' : optimizer.state_dict()
                                }, os.path.join(model_dir,"model_best.pth"))

                print("[Ep %d it %d\t PSNR : %.4f\t] ----  [best_Ep %d best_it %d Best_PSNR %.4f] " % (epoch, i, psnr_val_rgb,best_epoch,best_iter,best_psnr))

            model_restoration.train()

    scheduler.step()

    print("------------------------------------------------------------------")
    print("Epoch: {}\tTime: {:.4f}\tLoss: {:.4f}\tLearningRate {:.6f}".format(epoch, time.time()-epoch_start_time,epoch_loss, scheduler.get_lr()[0]))
    print("------------------------------------------------------------------")

    torch.save({'epoch': epoch,
                'state_dict': model_restoration.state_dict(),
                'optimizer' : optimizer.state_dict()
                }, os.path.join(model_dir,"model_latest.pth"))

    torch.save({'epoch': epoch,
                'state_dict': model_restoration.state_dict(),
                'optimizer' : optimizer.state_dict()
                }, os.path.join(model_dir,f"model_epoch_{epoch}.pth"))

