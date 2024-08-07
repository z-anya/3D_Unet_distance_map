from scipy.spatial.distance import euclidean

from dataset.dataset_lits_val_origin import Val_Dataset
from dataset.dataset_lits_train_origin import Train_Dataset

from torch.utils.data import DataLoader
import torch
import torch.optim as optim
from tqdm import tqdm
import config

from models import UNet, ResUNet, KiUNet_min, SegNet

from utils import logger, weights_init, metrics, common, loss

import os
import numpy as np
from collections import OrderedDict

import matplotlib.pyplot as plt

import torch.nn.functional as F

def calculate_distance(coords1, coords2):
    return torch.norm(coords1 - coords2)

def get_coords_mean(mask):
    coords = np.transpose(np.nonzero(mask))
    if len(coords) == 0:
        print('a')
    return torch.tensor(np.mean(coords, axis=0), dtype=torch.float32)



def val(model, val_loader, loss_func, n_labels):
    model.eval()
    val_loss = metrics.LossAverage()
    val_dice = metrics.DiceAverage(n_labels)
    with torch.no_grad():
        for idx, (data, target, bone) in tqdm(enumerate(val_loader), total=len(val_loader)):
            data, target, bone = data.float(), target.long(), bone.long()
            target = common.to_one_hot_3d(target, n_labels)
            data, target, bone = data.to(device), target.to(device), bone.to(device)
            output = model(data)
            loss = loss_func(output, target)

            val_loss.update(loss.item(), data.size(0))
            val_dice.update(output, target)
    val_log = OrderedDict({'Val_Loss': val_loss.avg, 'Val_dice_liver': val_dice.avg[1]})
    if n_labels == 3: val_log.update({'Val_dice_tumor': val_dice.avg[2]})
    return val_log


def train(model, train_loader, optimizer, loss_func, n_labels, alpha):
    print("=======Epoch:{}=======lr:{}".format(epoch, optimizer.state_dict()['param_groups'][0]['lr']))
    model.train()
    train_loss = metrics.LossAverage()
    train_dice = metrics.DiceAverage(n_labels)

    for idx, (data, target, bone) in tqdm(enumerate(train_loader), total=len(train_loader)):
        data, target, bone = data.float(), target.long(), bone.long()
        target = common.to_one_hot_3d(target, n_labels)
        data, target, bone = data.to(device), target.to(device), bone.to(device)
        optimizer.zero_grad()

        output = model(data)

        # 计算colon预测图的坐标均值
        output_coords_mean = get_coords_mean(output[3].detach().cpu().numpy())

        # 计算bone的坐标均值
        bone_coords_mean = get_coords_mean(bone.detach().cpu().numpy())

        # 计算colon的Ground Truth的坐标均值
        mask_coords_mean = get_coords_mean(target.detach().cpu().numpy())

        # 计算distance1和distance2
        distance1 = F.pairwise_distance(output_coords_mean, bone_coords_mean)
        distance2 = F.pairwise_distance(mask_coords_mean, bone_coords_mean)

        loss0 = loss_func(output[0], target)
        loss1 = loss_func(output[1], target)
        loss2 = loss_func(output[2], target)
        loss3 = loss_func(output[3], target)

        loss = loss3 + alpha * (loss0 + loss1 + loss2)

        loss_distance = torch.abs(distance1 - distance2)
        loss = loss + loss_distance

        loss.backward()
        optimizer.step()

        train_loss.update(loss3.item(), data.size(0))
        train_dice.update(output[3], target)

    val_log = OrderedDict({'Train_Loss': train_loss.avg, 'Train_dice_liver': train_dice.avg[1]})
    if n_labels == 3: val_log.update({'Train_dice_tumor': train_dice.avg[2]})
    return val_log


if __name__ == '__main__':
    args = config.args
    save_path = os.path.join('./experiments', args.save)
    if not os.path.exists(save_path): os.mkdir(save_path)
    device = torch.device('cpu' if args.cpu else 'cuda')
    # data info
    train_loader = DataLoader(dataset=Train_Dataset(args), batch_size=args.batch_size, num_workers=args.n_threads,
                              shuffle=True)
    val_loader = DataLoader(dataset=Val_Dataset(args), batch_size=1, num_workers=args.n_threads, shuffle=False)

    # model info
    model = ResUNet(in_channel=1, out_channel=args.n_labels, training=True).to(device)

    model.apply(weights_init.init_model)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    common.print_network(model)
    # model = torch.nn.DataParallel(model, device_ids=args.gpu_id)  # multi-GPU

    loss_func = loss.TverskyLoss()
    log = logger.Train_Logger(save_path, "train_log")

    best = [0, 0]  # 初始化最优模型的epoch和performance
    trigger = 0  # early stop 计数器
    alpha = 0.4  # 深监督衰减系数初始值

    val_dice_values = []  # 存储每个 epoch 的 val Dice 值
    train_dice_values = []  ## 存储每个 epoch 的 train Dice 值
    val_loss = []
    train_loss = []

    for epoch in range(1, args.epochs + 1):
        common.adjust_learning_rate(optimizer, epoch, args)
        train_log = train(model, train_loader, optimizer, loss_func, args.n_labels, alpha)
        val_log = val(model, val_loader, loss_func, args.n_labels)
        log.update(epoch, train_log, val_log)

        val_dice_values.append(val_log['Val_dice_liver'])  # 如果有多个类别，可以选择其他 Dice 值
        train_dice_values.append(train_log['Train_dice_liver'])
        val_loss.append(val_log['Val_Loss'])
        train_loss.append(train_log['Train_Loss'])

        # Save checkpoint.
        state = {'net': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}
        torch.save(state, os.path.join(save_path, 'latest_model.pth'))
        trigger += 1
        if val_log['Val_dice_liver'] > best[1]:
            print('Saving best model')
            torch.save(state, os.path.join(save_path, 'best_model.pth'))
            best[0] = epoch
            best[1] = val_log['Val_dice_liver']
            trigger = 0
        print('Best performance at Epoch: {} | {}'.format(best[0], best[1]))

        # 深监督系数衰减
        if epoch % 30 == 0: alpha *= 0.8

        # # early stopping
        # if args.early_stop is not None:
        #     if trigger >= args.early_stop:
        #         print("=> early stopping")
        #         break
        torch.cuda.empty_cache()

    plt.plot(range(1, args.epochs + 1), val_dice_values, marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('val Dice Value')
    plt.title('mask val dice')
    plt.grid(True)
    # plt.ion()  # 开启交互模式
    plt.show()

    plt.plot(range(1, args.epochs + 1), train_dice_values, marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('train Dice Value')
    plt.title('mask train dice')
    plt.grid(True)
    plt.show()

    plt.plot(range(1, args.epochs + 1), val_loss, marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('val loss')
    plt.title('mask val loss')
    plt.grid(True)
    plt.show()

    plt.plot(range(1, args.epochs + 1), train_loss, marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('train loss')
    plt.title('mask train loss')
    plt.grid(True)
    plt.show()