from dataset.dataset_lits_val import Val_Dataset
from dataset.dataset_lits_train import Train_Dataset

from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import config

from models import UNet, ResUNet, KiUNet_min, SegNet
from models.dsResUNet import ResUNetDS

from utils import logger, weights_init, metrics, common, loss
import os
import numpy as np
import nibabel as nib

from collections import OrderedDict

import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score


def val(model, val_loader, loss_func, n_labels):
    model.eval()
    val_loss = metrics.LossAverage()
    val_dice = metrics.DiceAverage(n_labels)

    # Initialize lists to store evaluation metrics
    val_recall = []
    val_f1 = []
    val_precision = []

    with torch.no_grad():
        for idx, (data, filter, mask) in tqdm(enumerate(val_loader), total=len(val_loader)):
            data, filter, mask = data.float(), filter.float(), mask.long()
            mask = common.to_one_hot_3d(mask, n_labels)
            data, filter, mask = data.to(device), filter.to(device), mask.to(device)
            output = model(data)

            loss = loss_func(output, mask)
            val_loss.update(loss.item(), data.size(0))
            val_dice.update(output, mask)

            # Convert the output and target to numpy arrays
            output_np = output.argmax(dim=1).cpu().numpy()
            mask_np = mask.argmax(dim=1).cpu().numpy()

            # Calculate confusion matrix
            conf_matrix = confusion_matrix(mask_np.ravel(), output_np.ravel())
            tn, fp, fn, tp = conf_matrix.ravel()

            # Calculate evaluation metrics
            accuracy = accuracy_score(mask_np.ravel(), output_np.ravel())
            precision = precision_score(mask_np.ravel(), output_np.ravel())
            recall = recall_score(mask_np.ravel(), output_np.ravel())
            f1 = f1_score(mask_np.ravel(), output_np.ravel())

            val_recall.append(recall)
            val_f1.append(f1)
            val_precision.append(precision)
    # Calculate average recall, F1 score, and precision
    avg_recall = np.mean(val_recall)
    avg_f1 = np.mean(val_f1)
    avg_precision = np.mean(val_precision)

    val_log = OrderedDict(
        {'Val_Loss': val_loss.avg, 'val_dice': val_dice.avg[1], 'Val_Recall': avg_recall, 'Val_F1': avg_f1,
         'Val_Precision': avg_precision})
    if n_labels == 3: val_log.update({'Val_dice_tumor': val_dice.avg[2]})
    return val_log


def train(model, train_loader, optimizer, loss_func, n_labels, alpha):
    print("=======Epoch:{}=======lr:{}".format(epoch, optimizer.state_dict()['param_groups'][0]['lr']))
    model.train()
    train_loss = metrics.LossAverage()
    train_dice = metrics.DiceAverage(n_labels)

    for idx, (data, filter, mask) in tqdm(enumerate(train_loader), total=len(train_loader)):
        data, filter, mask = data.float(), filter.float(), mask.long()
        mask = common.to_one_hot_3d(mask, n_labels)
        data, filter, mask = data.to(device), filter.to(device), mask.to(device)

        # # 查看第一个样本，第一个通道，第60个切片
        # slice_60_data = data[0, 0, 59, :, :]
        # slice_60_mask_0 = target[0, 0, 59, :, :]
        # # slice_60_mask_1 = target[0, 1, 59, :, :]
        #
        # # 将张量从GPU移到CPU，并转换为NumPy数组
        # slice_60_data = slice_60_data.cpu().numpy()
        # slice_60_mask_0 = slice_60_mask_0.cpu().numpy()
        # # slice_60_mask_1 = slice_60_mask_1.cpu().numpy()
        #
        # # 使用 matplotlib 查看这个切片
        # import matplotlib.pyplot as plt
        #
        # plt.imshow(slice_60_data, cmap='gray')
        # plt.show()
        # plt.imshow(slice_60_mask_0, cmap='gray')
        # plt.show()
        # plt.imshow(slice_60_mask_1, cmap='gray')
        # plt.show()
        optimizer.zero_grad()

        output = model(data)

        loss0 = loss_func(output[0], mask)
        loss1 = loss_func(output[1], mask)
        loss2 = loss_func(output[2], mask)
        loss3 = loss_func(output[3], mask)

        loss = loss3 + alpha * (loss0 + loss1 + loss2)
        loss.backward()
        optimizer.step()

        train_loss.update(loss.item(), data.size(0))
        train_dice.update(output[3], mask)

        # # 保存网络的每一层输出为NIfTI文件
        # temp = target.squeeze()
        # target_int16 = temp[1].cpu().detach().numpy()#.astype(np.int16)
        # output_nii_target = nib.Nifti1Image(target_int16, np.eye(4)) # 如果其中的output[3]是一个五维张量，五个维度分别是
        # output_dir = os.path.join(save_path, f'epoch_{epoch}', f'patient_{idx}')
        # os.makedirs(output_dir, exist_ok=True)
        # output_filename = os.path.join(output_dir, f'output_{epoch}_{idx}_target.nii')
        # nib.save(output_nii_target, output_filename)
        #
        # temp = output[3].squeeze()
        # mask_int16 = temp[1].cpu().detach().numpy()#.astype(np.int16)
        # output_nii = nib.Nifti1Image(mask_int16, np.eye(4))#如果其中的output[3]是一个五维张量，五个维度分别是
        # # for x, i in enumerate(mask_int16):
        # #     for y, j in enumerate(i):
        # #         for z, k in enumerate(j):
        # #             if int(k) != 0:
        # #                 print(f"({x}, {y}, {z}) = {int(k)}")
        # output_dir = os.path.join(save_path, f'epoch_{epoch}', f'patient_{idx}')
        # os.makedirs(output_dir, exist_ok=True)
        # output_filename = os.path.join(output_dir, f'output_{epoch}_{idx}.nii')
        # nib.save(output_nii, output_filename)

    train_log = OrderedDict({'Train_Loss': train_loss.avg, 'train_dice': train_dice.avg[1]})
    if n_labels == 3: train_log.update({'Train_dice_tumor': train_dice.avg[2]})
    return train_log


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
    model = ResUNetDS(in_channel=1, out_channel=args.n_labels, training=True).to(device)

    model.apply(weights_init.init_model)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    common.print_network(model)
    # model = torch.nn.DataParallel(model, device_ids=args.gpu_id)  # multi-GPU

    loss = loss.TverskyLoss()
    # loss = nn.BCEWithLogitsLoss()
    # loss = loss.DiceLoss()

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
        # if epoch == 3:
        #     print(1)
        train_log = train(model, train_loader, optimizer, loss, args.n_labels, alpha)
        val_log = val(model, val_loader, loss, args.n_labels)
        log.update(epoch, train_log, val_log)

        val_dice_values.append(val_log['val_dice'])  # 如果有多个类别，可以选择其他 Dice 值
        train_dice_values.append(train_log['train_dice'])
        val_loss.append(val_log['Val_Loss'])
        train_loss.append(train_log['Train_Loss'])

        # Save checkpoint.
        state = {'net': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}
        torch.save(state, os.path.join(save_path, 'latest_model.pth'))
        trigger += 1
        if val_log['val_dice'] > best[1]:
            print('Saving best model')
            torch.save(state, os.path.join(save_path, 'best_model.pth'))
            best[0] = epoch
            best[1] = val_log['val_dice']
            trigger = 0
        print('Best performance at Epoch: {} | {}'.format(best[0], best[1]))

        # 深监督系数衰减
        if epoch % 30 == 0: alpha *= 0.8

        # early stopping
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

