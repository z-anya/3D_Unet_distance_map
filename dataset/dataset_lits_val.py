from posixpath import join
from torch.utils.data import DataLoader
import os
import sys
import numpy as np
import SimpleITK as sitk
import torch
from torch.utils.data import Dataset as dataset
from .transforms import Center_Crop, Compose


class Val_Dataset(dataset):
    def __init__(self, args, is_test=False):

        self.args = args
        if not is_test:
            self.filename_list = self.load_file_name_list(os.path.join(args.dataset_path, 'val_path_list.txt'))
        else:
            self.filename_list = self.load_file_name_list(os.path.join(args.dataset_path, 'test_path_list.txt'))

        self.transforms = Compose([Center_Crop(base=16, max_size=args.val_crop_max_size)])

    def __getitem__(self, index):

        ct = sitk.ReadImage(self.filename_list[index][0] )
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
            ct_array, seg_array, cl_array, dm_array = self.transforms(ct_array, seg_array, cl_array, dm_array)

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
    sys.path.append('/ssd/lzq/3DUNet')
    from config import args
    train_ds = Val_Dataset(args)

    # 定义数据加载
    train_dl = DataLoader(train_ds, 2, False, num_workers=1)

    for i, (ct, seg, bone) in enumerate(train_dl):
        print(i, ct.size(), seg.size(), bone.size())