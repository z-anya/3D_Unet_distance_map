from collections import Counter
from posixpath import join
from torch.utils.data import DataLoader
import os
import sys
import random
from torchvision.transforms import RandomCrop
import numpy as np
import SimpleITK as sitk
import torch
from torch.utils.data import Dataset as dataset
from .transforms import RandomCrop, RandomFlip_LR, RandomFlip_UD, Center_Crop, Compose, Resize

class Train_Dataset(dataset):
    def __init__(self, args):

        self.args = args

        self.filename_list = self.load_file_name_list(os.path.join(args.dataset_path, 'train_path_list.txt'))

        self.transforms = Compose([
                RandomCrop(self.args.crop_size),
                RandomFlip_LR(prob=0.5),
                RandomFlip_UD(prob=0.5),
                # RandomRotate()
            ])

    def __getitem__(self, index):

        ct = sitk.ReadImage(self.filename_list[index][0])
        seg = sitk.ReadImage(self.filename_list[index][1])
        cl = sitk.ReadImage(self.filename_list[index][2])
        dm = sitk.ReadImage(self.filename_list[index][3])

        ct_array = sitk.GetArrayFromImage(ct)
        seg_array = sitk.GetArrayFromImage(seg)
        cl_array = sitk.GetArrayFromImage(cl)
        dm_array = sitk.GetArrayFromImage(dm)
        # print(ct_array.shape, seg_array.shape)
        ct_array = ct_array / self.args.norm_factor
        ct_array = ct_array.astype(np.float32)
        ct_array = torch.FloatTensor(ct_array).unsqueeze(0)
        seg_array = torch.FloatTensor(seg_array).unsqueeze(0)
        cl_array = torch.FloatTensor(cl_array).unsqueeze(0)
        dm_array = torch.FloatTensor(dm_array).unsqueeze(0)

        if self.transforms:
            # ct_array, seg_array, bone_array = self.transforms(ct_array, seg_array, bone_array)
            ct_array, seg_array, cl_array, dm_array = self.transforms(ct_array, seg_array, cl_array, dm_array)

        # print(ct_array.shape, seg_array.shape, bone_array.shape)


        # return ct_array, seg_array.squeeze(0), bone_array
        return ct_array, seg_array, cl_array, dm_array

    def __len__(self):
        return len(self.filename_list)

    def load_file_name_list(self, file_path):
        file_name_list = []
        with open(file_path, 'r') as file_to_read:
            while True:
                lines = file_to_read.readline().strip()  # 整行读取数据
                if not lines:
                    break
                file_name_list.append(lines.split())
        return file_name_list

if __name__ == "__main__":
    sys.path.append('/home/dalhxwlyjsuo/guest_lizg/unet')
    from config import args
    train_ds = Train_Dataset(args)

    # 定义数据加载
    train_dl = DataLoader(train_ds, 2, False, num_workers=1)
    from models.zyx_Unet import get_args, UnitedNet

    args = get_args()
    print(f"加载模型")
    model = UnitedNet(args)
    import pdb
    # pdb.set_trace()
    for i, (ct, seg, cl, dm) in enumerate(train_dl):
        print(f"{i}")
        pdb.set_trace()
        output = model(ct)