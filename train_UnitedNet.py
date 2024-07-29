import argparse
import csv
import glob
import os
from typing import Mapping, Any, Iterator

import math
import pandas as pd
import torch.nn as nn
import cv2
import numpy as np
from PIL import Image
import torch
from deepspeed.runtime.lr_schedules import WarmupCosineLR
from einops import einops
from natsort import natsort, natsorted
from peft import LoraConfig, get_peft_model
from torch.utils.data import random_split, DataLoader, DistributedSampler, IterableDataset, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel, LlamaForCausalLM, \
    ImageGPTForCausalImageModeling, ImageGPTImageProcessor
import deepspeed
from deepspeed.ops.adam import FusedAdam
from tqdm import tqdm
import pdb

from dataset.dataset_lits_test import Img_DataSet, Test_Datasets
from utils import loss as loss_module
from transformers import PreTrainedModel, PretrainedConfig
from torch.nn import functional as F
import wandb
import time
from config import args

from dataset.dataset_lits_train import Train_Dataset
from dataset.dataset_lits_val import Val_Dataset
from models.zyx_Unet import UnitedNet, ZyxUNet
from utils.common import to_one_hot_3d
from utils.metrics import DiceAverage


def get_train_ds_config(train_batch_size,
                        micro_batch_size_per_gpu,
                        gradient_accumulation_steps,
                        world_size,
                        dtype="bf16",
                        stage=1,
                        offload=None,
                        enable_tensorboard=False,
                        tb_path="",
                        tb_name=""
                        ):
    print(f"train_batch_size:{train_batch_size}, world_size:{world_size}, "
          f"gradient_accumulation_steps:{gradient_accumulation_steps}, micro_batch_size_per_gpu:{micro_batch_size_per_gpu}")
    assert train_batch_size == world_size * gradient_accumulation_steps * micro_batch_size_per_gpu
    device = offload if offload else "none"
    if dtype == "fp16":
        data_type = "fp16"
        dtype_config = {"enabled": True, "loss_scale_window": 100}
    elif dtype == "bf16":
        data_type = "bfloat16"
        dtype_config = {"enabled": True}
    elif dtype == "fp32":
        data_type = "fp32"
        dtype_config = {"enabled": True}
    zero_opt_dict = {
        "stage": stage,
        "contiguous_gradients": True,
        "stage3_max_live_parameters": 1e9,
        "stage3_max_reuse_distance": 1e9,
        "stage3_prefetch_bucket_size": 1e7,
        "stage3_param_persistence_threshold": 1e5,
        "reduce_bucket_size": 1e7,
        "sub_group_size": 1e9,
        "offload_optimizer": {
            "device": device
        },
        "offload_param": {
            "device": device
        }
    }
    return {
        "activation_checkpointing": {
            "partition_activations": True,
            "contiguous_memory_optimization": True,
        },
        "train_batch_size": train_batch_size,
        "train_micro_batch_size_per_gpu": micro_batch_size_per_gpu,
        "steps_per_print": 10,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "zero_optimization": zero_opt_dict,
        data_type: dtype_config,
        "gradient_clipping": 1.0,
        "prescale_gradients": True,
        "wall_clock_breakdown": False,
        "tensorboard": {
            "enabled": enable_tensorboard,
            "output_path": f"{tb_path}/ds_tensorboard_logs/",
            "job_name": f"{tb_name}_tensorboard"
        }
    }
def get_torch_float_dtype(dtype):
    if dtype in (torch.float16, torch.bfloat16, torch.float32):
        return dtype
    return {
        'float16': torch.float16,
        'fp16': torch.float16,
        'f16': torch.float16,
        'bfloat16': torch.bfloat16,
        'bf16': torch.bfloat16,
        'float32': torch.float32,
        'fp32': torch.float32,
        'f32': torch.float32,
    }[dtype]


def get_model(config, checkpoint_path=None):
    if checkpoint_path is None:
        seg_unet = ZyxUNet(config, True)
        # center line unet
        cl_unet = ZyxUNet(config, False)
        # distance map unet
        dm_unet = ZyxUNet(config, False)
    else:
        seg_unet = ZyxUNet(config, True)
        cl_unet = ZyxUNet(config, False)
        dm_unet = ZyxUNet(config, False)

        seg_unet_path = os.path.join(checkpoint_path, 'seg', 'seg_unet_last', 'mp_rank_00_model_states.pt')
        cl_unet_path = os.path.join(checkpoint_path, 'cl', 'cl_unet_last', 'mp_rank_00_model_states.pt')
        dm_unet_path = os.path.join(checkpoint_path, 'dm', 'dm_unet_last', 'mp_rank_00_model_states.pt')

        seg_state_dict = torch.load(seg_unet_path)['module']
        cl_state_dict = torch.load(cl_unet_path)['module']
        dm_state_dict = torch.load(dm_unet_path)['module']

        seg_unet.load_state_dict(seg_state_dict)
        cl_unet.load_state_dict(cl_state_dict)
        dm_unet.load_state_dict(dm_state_dict)

    return seg_unet, cl_unet, dm_unet

def train_main(args):
    if args.local_rank == -1:
        deepspeed.init_distributed()
    else:
        print(f"当前设备：{args.local_rank}")
        # get_accelerator().set_device(args.local_rank)
        deepspeed.init_distributed()
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device("cuda", local_rank)
    seg_unet, cl_unet, dm_unet = get_model(args)
    train_dataset = Train_Dataset(args)
    val_dataset = Val_Dataset(args)
    train_length = len(train_dataset)
    val_length = len(val_dataset)

    train_sampler = DistributedSampler(train_dataset)
    val_sampler = DistributedSampler(val_dataset)
    train_dataloader = DataLoader(train_dataset, batch_size=args.micro_batch_size_per_gpu, sampler=train_sampler)
    val_dataloader = DataLoader(val_dataset,  batch_size=args.micro_batch_size_per_gpu, sampler=val_sampler)
    seg_optimizer = FusedAdam(get_train_parameters(seg_unet),
                          weight_decay=args.weight_decay,
                          lr=args.lr,
                          betas=(0.9, 0.95))
    cl_optimizer = FusedAdam(get_train_parameters(cl_unet),
                              weight_decay=args.weight_decay,
                              lr=args.lr,
                              betas=(0.9, 0.95))
    dm_optimizer = FusedAdam(get_train_parameters(dm_unet),
                              weight_decay=args.weight_decay,
                              lr=args.lr,
                              betas=(0.9, 0.95))

    data_length = len(train_dataloader)
    update_steps_per_epoch = math.ceil(data_length / args.gradient_accumulation_steps)
    total_steps = update_steps_per_epoch * args.epochs
    seg_lr_scheduler = WarmupCosineLR(optimizer=seg_optimizer,
                                  total_num_steps=total_steps,
                                  warmup_num_steps=args.warmup_steps)
    cl_lr_scheduler = WarmupCosineLR(optimizer=cl_optimizer,
                                      total_num_steps=total_steps,
                                      warmup_num_steps=args.warmup_steps)
    dm_lr_scheduler = WarmupCosineLR(optimizer=dm_optimizer,
                                      total_num_steps=total_steps,
                                      warmup_num_steps=args.warmup_steps)
    ds_config = get_train_ds_config(args.batch_size,
                                    args.micro_batch_size_per_gpu,
                                    args.gradient_accumulation_steps,
                                    args.world_size,
                                    stage=args.ds_stage,
                                    dtype='fp32')
    seg_model, seg_optimizer, _, seg_lr_scheduler, = deepspeed.initialize(model=seg_unet,
                                                              optimizer=seg_optimizer,
                                                              config=ds_config,
                                                              lr_scheduler=seg_lr_scheduler
                                                                )
    cl_model, cl_optimizer, _, cl_lr_scheduler, = deepspeed.initialize(model=cl_unet,
                                                                  optimizer=cl_optimizer,
                                                                  config=ds_config,
                                                                  lr_scheduler=cl_lr_scheduler
                                                                  )
    dm_model, dm_optimizer, _, dm_lr_scheduler, = deepspeed.initialize(model=dm_unet,
                                                                    optimizer=dm_optimizer,
                                                                    config=ds_config,
                                                                    lr_scheduler=dm_lr_scheduler
                                                                    )
    # assert args.is_resume in [0, 1]
    # if args.is_resume == 1:
    #     load_path, client_state = model.load_checkpoint(args.checkpoint_path, tag=args.checkpoint_tag)
    #     print(f"load checkpoint from {load_path}")
    current = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    wandb.init(project="Unet",
               name=current,
               config=args)
    dice_loss = loss_module.DiceLoss()
    mse_loss = torch.nn.MSELoss()
    loss_func = {'dice_loss': dice_loss, 'mse_loss': mse_loss}
    for epoch in range(args.epochs):
        train_one_epoch(seg_model, cl_model, dm_model, train_dataloader, val_dataloader, epoch, device, loss_func)
        if epoch % 50 == 0:
            seg_model.save_checkpoint("checkpoints/seg", tag=f"seg_unet{epoch}")
            cl_model.save_checkpoint("checkpoints/cl", tag=f"cl_unet{epoch}")
            dm_model.save_checkpoint("checkpoints/dm", tag=f"dm_unet{epoch}")
    seg_model.save_checkpoint("checkpoints/seg", tag=f"seg_unet_last")
    cl_model.save_checkpoint("checkpoints/cl", tag=f"cl_unet_last")
    dm_model.save_checkpoint("checkpoints/dm", tag=f"dm_unet_last")

def train_one_epoch(seg_model,
                    cl_model,
                    dm_model,
                    train_dataloader, val_dataloader, epoch, device, loss_func):
    with tqdm(range(len(train_dataloader))) as pbar:
        for i, (ct, seg, cl, dm) in zip(pbar, train_dataloader):
            ct = ct.to(device)
            seg = seg.to(device)
            cl = cl.to(device)
            dm = dm.to(device)
            # encoder
            cl_maps, cl_mid = cl_model.encoder(ct)
            dm_maps, dm_mid = dm_model.encoder(ct)
            seg_maps, seg_mid = seg_model.encoder(ct, cl_maps)

            # attn block
            seg_mid = seg_model.attn_block(seg_mid, dm_mid)

            # decoder
            seg_output = seg_model.decoder(seg_mid, seg_maps)
            cl_output = cl_model.decoder(cl_mid, cl_maps)
            dm_output = dm_model.decoder(dm_mid, dm_maps)
            # loss
            dice_loss = loss_func['dice_loss']
            mse_loss = loss_func['mse_loss']
            seg_loss = dice_loss(seg_output, seg)
            cl_loss = dice_loss(cl_output, cl)
            dm_loss = mse_loss(dm_output, dm)

            # backward
            seg_model.backward(seg_loss, retain_graph=True)
            cl_model.backward(cl_loss)
            dm_model.backward(dm_loss)

            seg_model.step()
            cl_model.step()
            dm_model.step()

            lr = seg_model.get_lr()[0]
            wandb.log({"Seg Loss": seg_loss, "CL Loss": cl_loss, "DM Loss": dm_loss, "LR": lr,})
            pbar.set_postfix(seg_loss=seg_loss.detach().cpu().to(torch.float32).numpy(),
                             cl_loss=cl_loss.detach().cpu().to(torch.float32).numpy(),
                             dm_loss=dm_loss.detach().cpu().to(torch.float32).numpy(),
                             Epoch=epoch)
            del ct, seg, cl, dm, seg_loss, cl_loss, dm_loss
            torch.cuda.empty_cache()
            if i  % 1000 == 0:
                print(f"current step {i}")
                val_step(seg_model, cl_model, dm_model, val_dataloader, epoch, device, loss_func)



def val_step(seg_model,
            cl_model,
            dm_model,
            val_dataloader, epoch, device, loss_func):
    seg_model.eval()
    cl_model.eval()
    dm_model.eval()
    eval_steps = 0
    eval_seg_loss = 0
    eval_cl_loss = 0
    eval_dm_loss = 0
    with tqdm(range(len(val_dataloader))) as pbar:
        for i, (ct, seg, cl, dm) in zip(pbar, val_dataloader):
            with torch.no_grad():
                ct = ct.to(device)
                seg = seg.to(device)
                cl = cl.to(device)
                dm = dm.to(device)
                # encoder
                cl_maps, cl_mid = cl_model.encoder(ct)
                dm_maps, dm_mid = dm_model.encoder(ct)
                seg_maps, seg_mid = seg_model.encoder(ct, cl_maps)

                # attn block
                seg_mid = seg_model.attn_block(seg_mid, dm_mid)

                # decoder
                seg_output = seg_model.decoder(seg_mid, seg_maps)
                cl_output = cl_model.decoder(cl_mid, cl_maps)
                dm_output = dm_model.decoder(dm_mid, dm_maps)
                # loss
                dice_loss = loss_func['dice_loss']
                mse_loss = loss_func['mse_loss']
                seg_loss = dice_loss(seg_output, seg)
                cl_loss = dice_loss(cl_output, cl)
                dm_loss = mse_loss(dm_output, dm)
            # batch shape: (b, t, c, h, w) t is length of frame sequence
            eval_seg_loss += seg_loss
            eval_cl_loss += cl_loss
            eval_dm_loss += dm_loss
            eval_steps += 1
            del seg_output, cl_output, dm_output, seg_loss, cl_loss, dm_loss
            torch.cuda.empty_cache()
    wandb.log({"val_seg_loss": eval_seg_loss / eval_steps,
               "val_cl_loss": eval_cl_loss / eval_steps,
               "val_dm_loss": eval_dm_loss / eval_steps,})
    seg_model.train()
    cl_model.train()
    dm_model.train()


def test_main(args):
    if args.local_rank == -1:
        deepspeed.init_distributed()
    else:
        print(f"当前设备：{args.local_rank}")
        # get_accelerator().set_device(args.local_rank)
        deepspeed.init_distributed()
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device("cuda", local_rank)
    seg_unet, cl_unet, dm_unet = get_model(args, args.checkpoint)
    data_path = '/home/dalhxwlyjsuo/guest_lizg/data/nii_data/WORD/WORD-V0.1.0/imagesVal'
    label_path = '/home/dalhxwlyjsuo/guest_lizg/data/nii_data/WORD/WORD-V0.1.0/labelsVal'
    test_dataset = Test_Datasets(data_path, label_path, args)

    for dataset, file_idx in test_dataset:
        test_one(seg_unet, cl_unet, dm_unet, dataset, device, args)

def test_new(args):
    if args.local_rank == -1:
        deepspeed.init_distributed()
    else:
        print(f"当前设备：{args.local_rank}")
        # get_accelerator().set_device(args.local_rank)
        deepspeed.init_distributed()
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device("cuda", local_rank)
    seg_unet, cl_unet, dm_unet = get_model(args, args.checkpoint)
    val_dataset = Val_Dataset(args)
    val_sampler = DistributedSampler(val_dataset)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, sampler=val_sampler)
    dice_metrics = DiceAverage(2)
    loss_func = loss_module.DiceLoss()

    seg_model = deepspeed.init_inference(seg_unet, dtype=torch.float32)
    cl_model = deepspeed.init_inference(cl_unet, dtype=torch.float32)
    dm_model = deepspeed.init_inference(dm_unet, dtype=torch.float32)

    seg_model.eval()
    cl_model.eval()
    dm_model.eval()
    eval_steps = 0
    eval_seg_loss = 0
    eval_cl_loss = 0
    eval_dm_loss = 0
    with tqdm(range(len(val_dataloader))) as pbar:
        for i, (ct, seg, cl, dm) in zip(pbar, val_dataloader):
            with torch.no_grad():
                ct = ct.to(device)
                seg = seg.to(device)
                cl = cl.to(device)
                dm = dm.to(device)
                # encoder
                cl_maps, cl_mid = cl_model.module.encoder(ct)
                dm_maps, dm_mid = dm_model.module.encoder(ct)
                seg_maps, seg_mid = seg_model.module.encoder(ct, cl_maps)

                # attn block
                seg_mid = seg_model.module.attn_block(seg_mid, dm_mid)

                # decoder
                seg_output = seg_model.module.decoder(seg_mid, seg_maps)
                cl_output = cl_model.module.decoder(cl_mid, cl_maps)
                dm_output = dm_model.module.decoder(dm_mid, dm_maps)

                seg_loss = loss_func(seg_output, seg)
                cl_loss = loss_func(cl_output, cl)
                dm_loss = loss_func(dm_output, dm)
                eval_seg_loss += seg_loss
                eval_cl_loss += cl_loss
                eval_dm_loss += dm_loss
                eval_steps += 1
                dice_metrics.update(seg_output, seg)
            del seg_output, cl_output, dm_output,
            torch.cuda.empty_cache()
    print({"val_seg_loss": eval_seg_loss / eval_steps,
           "val_cl_loss": eval_cl_loss / eval_steps,
           "val_dm_loss": eval_dm_loss / eval_steps})
    print(dice_metrics.avg)


def test_one(seg_unet, cl_unet, dm_unet, test_dataset, device, args):
    # val_sampler = DistributedSampler(test_dataset)
    val_dataloader = DataLoader(dataset=test_dataset, batch_size=1)
    dice_metrics = DiceAverage(args.n_labels)
    loss_func = loss_module.DiceLoss()

    seg_model = deepspeed.init_inference(seg_unet, dtype=torch.float32)
    cl_model = deepspeed.init_inference(cl_unet, dtype=torch.float32)
    dm_model = deepspeed.init_inference(dm_unet, dtype=torch.float32)

    seg_model.eval()
    cl_model.eval()
    dm_model.eval()
    # eval_steps = 0
    # eval_seg_loss = 0
    # eval_cl_loss = 0
    # eval_dm_loss = 0
    with tqdm(range(len(val_dataloader))) as pbar:
        for i, (ct, seg,) in zip(pbar, val_dataloader):
            with torch.no_grad():
                ct = ct.to(device)
                # pdb.set_trace()
                seg = to_one_hot_3d(seg, args.n_labels)
                seg = seg.to(device)
                print(ct.shape)
                print(seg.shape)
                # cl = cl.to(device)
                # dm = dm.to(device)
                # encoder
                cl_maps, cl_mid = cl_model.module.encoder(ct)
                dm_maps, dm_mid = dm_model.module.encoder(ct)
                seg_maps, seg_mid = seg_model.module.encoder(ct, cl_maps)

                # attn block
                seg_mid = seg_model.module.attn_block(seg_mid, dm_mid)

                # decoder
                seg_output = seg_model.module.decoder(seg_mid, seg_maps)
                cl_output = cl_model.module.decoder(cl_mid, cl_maps)
                dm_output = dm_model.module.decoder(dm_mid, dm_maps)

                # seg_loss = loss_func(seg_output, seg)
                # cl_loss = loss_func(cl_output, cl)
                # dm_loss = loss_func(dm_output, dm)
                # eval_seg_loss += seg_loss
                # eval_cl_loss += cl_loss
                # eval_dm_loss += dm_loss
                # eval_steps += 1

                dice_metrics.update(seg_output, seg)
            del seg_output, cl_output, dm_output,
            torch.cuda.empty_cache()
    # print({"val_seg_loss": eval_seg_loss / eval_steps,
    #        "val_cl_loss": eval_cl_loss / eval_steps,
    #        "val_dm_loss": eval_dm_loss / eval_steps})

    print(dice_metrics.avg)


def get_train_parameters(model):
    # 可以自定义
    return model.parameters()


if __name__ == '__main__':
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    os.environ['TORCH_USE_CUDA_DSA'] = "1"
    os.environ["WANDB_API_KEY"] = "92f7db817e758b1939a5f7fa871f74dbeabbb0b1"
    os.environ["WANDB_MODE"] = "offline"
    wandb.login(key="92f7db817e758b1939a5f7fa871f74dbeabbb0b1")
    train_main(args)

    # test_main(args)
    # test_new(args)