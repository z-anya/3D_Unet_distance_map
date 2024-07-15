import glob
import pdb

import numpy as np
import os
import SimpleITK as sitk
import random

from natsort import natsort
from scipy import ndimage
from os.path import join
import config
import glob
import os

# class LITS_preprocess:
#     def __init__(self, raw_dataset_path, fixed_dataset_path, args):
#         self.raw_root_path = raw_dataset_path
#         self.fixed_path = fixed_dataset_path
#         self.classes = args.n_labels  # 分割类别数（只分割肝脏为2，或者分割肝脏和肿瘤为3）
#         self.upper = args.upper
#         self.lower = args.lower
#         self.expand_slice = args.expand_slice  # 轴向外侧扩张的slice数量
#         self.size = args.min_slices  # 取样的slice数量
#         self.xy_down_scale = args.xy_down_scale
#         self.slice_down_scale = args.slice_down_scale
#
#         self.valid_rate = args.valid_rate
#
#     def fix_data(self, data_root_pth, tag, is_label):
#         if not os.path.exists(self.fixed_path):  # 创建保存目录
#             os.makedirs(join(self.fixed_path, tag))
#         # ct_folder_path = os.path.join(self.raw_root_path, 'WORD', 'WORD-V0.1.0', 'imagesTr')  # 修改为ct文件夹所在的路径
#         # labels_folder_path = os.path.join(self.raw_root_path, 'WORD', 'label_bone_colon')
#
#         file_list = glob.glob(data_root_pth + '/*.nii.gz')
#         Numbers = len(file_list)
#         print('Total numbers of samples is :', Numbers)
#         for i, ct_file in enumerate(file_list):
#             print("==== {} | {}/{} ====".format(ct_file, i + 1, Numbers))
#             new_ct = self.process(ct_file, is_label, classes=self.classes)
#             if new_ct != None:
#                 ct_save_root_path = os.path.join(self.fixed_path, tag)
#                 if not os.path.exists(ct_save_root_path):
#                     os.makedirs(ct_save_root_path)
#                 save_path = os.path.join(ct_save_root_path, os.path.basename(ct_file))
#                 print(f"save path {save_path}")
#                 sitk.WriteImage(new_ct, save_path)
#
#     def process(self, ct_path, is_label, classes=None):
#         img = sitk.ReadImage(ct_path, sitk.sitkFloat64)
#         img_array = sitk.GetArrayFromImage(img)
#
#         print("Ori shape:", img_array.shape)
#
#         if classes==2:
#             # 将金标准中肝脏和肝肿瘤的标签融合为一个
#             img_array[img_array > 0] = 1
#
#         # 降采样，（对x和y轴进行降采样，slice轴的spacing归一化到slice_down_scale）
#         if is_label:
#             ct_array = ndimage.zoom(img_array,
#                                     (img.GetSpacing()[-1] / self.slice_down_scale, self.xy_down_scale,
#                                      self.xy_down_scale),
#                                     order=0)
#         else:
#             ct_array = ndimage.zoom(img_array,
#                                     (img.GetSpacing()[-1] / self.slice_down_scale, self.xy_down_scale, self.xy_down_scale),
#                                     order=3)
#         # 找到肝脏区域开始和结束的slice，并各向外扩张
#         pdb.set_trace()
#         z = np.any(img_array, axis=(1, 2))
#         start_slice, end_slice = np.where(z)[0][[0, -1]]
#
#         # 两个方向上各扩张个slice
#         if start_slice - self.expand_slice < 0:
#             start_slice = 0
#         else:
#             start_slice -= self.expand_slice
#
#         if end_slice + self.expand_slice >= img_array.shape[0]:
#             end_slice = img_array.shape[0] - 1
#         else:
#             end_slice += self.expand_slice
#
#         print("Cut out range:", str(start_slice) + '--' + str(end_slice))
#         # 如果这时候剩下的slice数量不足size，直接放弃，这样的数据很少
#         if end_slice - start_slice + 1 < self.size:
#             print('Too little slice，give up the sample:', ct_path)
#             return None
#         #截取保留区域
#         ct_array = ct_array[start_slice:end_slice + 1, :, :]
#
#         print("Preprocessed shape:", ct_array.shape)
#
#         # 保存为对应的格式
#         new_ct = sitk.GetImageFromArray(ct_array)
#         new_ct.SetDirection(img.GetDirection())
#         new_ct.SetOrigin(img.GetOrigin())
#         new_ct.SetSpacing((img.GetSpacing()[0] * int(1 / self.xy_down_scale),
#                            img.GetSpacing()[1] * int(1 / self.xy_down_scale), self.slice_down_scale))
#
#         return new_ct
#
#     def write_train_val_name_list(self):
#         data_name_list = os.listdir(join(self.fixed_path, "ct"))
#         data_num = len(data_name_list)
#         print('the fixed dataset total numbers of samples is :', data_num)
#         random.shuffle(data_name_list)
#
#         assert self.valid_rate < 1.0
#         train_name_list = data_name_list[0:int(data_num * (1 - self.valid_rate))]
#         val_name_list = data_name_list[
#                         int(data_num * (1 - self.valid_rate)):int(data_num * ((1 - self.valid_rate) + self.valid_rate))]
#
#         self.write_name_list(train_name_list, "train_path_list.txt")
#         self.write_name_list(val_name_list, "val_path_list.txt")
#
#     def write_name_list(self, name_list, file_name):
#         f = open(join(self.fixed_path, file_name), 'w')
#         for name in name_list:
#             ct_path = os.path.join(self.fixed_path, 'ct', name)
#             labels_path = os.path.join(self.fixed_path, 'labels', name)
#             f.write(ct_path + ' ' + labels_path + "\n")
#         f.close()


class LITS_preprocess:
    def __init__(self, raw_dataset_path, fixed_dataset_path, args):
        self.raw_root_path = raw_dataset_path
        self.fixed_path = fixed_dataset_path
        self.classes = args.n_labels  # 分割类别数（只分割肝脏为2，或者分割肝脏和肿瘤为3）
        self.upper = args.upper
        self.lower = args.lower
        self.expand_slice = args.expand_slice  # 轴向外侧扩张的slice数量
        self.size = args.min_slices  # 取样的slice数量
        self.xy_down_scale = args.xy_down_scale
        self.slice_down_scale = args.slice_down_scale

        self.valid_rate = args.valid_rate

    def fix_data(self, ct_path=None, seg_path=None, cl_path=None, dm_path=None):
      # 创建保存目录
        os.makedirs(join(self.fixed_path, 'ct'), exist_ok=True)
        os.makedirs(join(self.fixed_path, 'seg'), exist_ok=True)
        os.makedirs(join(self.fixed_path, 'cl'), exist_ok=True)
        os.makedirs(join(self.fixed_path, 'dm'), exist_ok=True)
        if ct_path is not None:
            ct_folder_path = ct_path
        else:
            ct_folder_path = os.path.join(self.raw_root_path, 'WORD', 'WORD-V0.1.0', 'imagesTr')

        if seg_path is not None:
            seg_folder_path = seg_path
        else:
            seg_folder_path = os.path.join(self.raw_root_path, 'WORD', 'label_bone_colon')

        if cl_path is not None:
            cl_folder_path = cl_path
        else:
            cl_folder_path = os.path.join(self.raw_root_path, 'WORD', 'label_bone_colon')

        if dm_path is not None:
            dm_folder_path = dm_path
        else:
            dm_folder_path = os.path.join(self.raw_root_path, 'WORD', 'label_bone_colon')

        ct_file_list = natsort.natsorted(os.listdir(ct_folder_path))
        seg_file_list = natsort.natsorted(os.listdir(seg_folder_path))
        cl_file_list = natsort.natsorted(os.listdir(cl_folder_path))
        dm_file_list =natsort.natsorted(os.listdir(dm_folder_path))
        Numbers = len(ct_file_list)
        print('Total numbers of samples is :', Numbers)
        for ct_file, seg_file, cl_file, dm_file, i in zip(ct_file_list, seg_file_list, cl_file_list, dm_file_list, range(Numbers)):
            print("==== {} | {}/{} ====".format(ct_file, i + 1, Numbers))
            # if "word_0049" not in ct_file:
            #     continue
            # pdb.set_trace()
            print(f"ct: {ct_file}, seg: {seg_file}, cl: {cl_file}, dm: {dm_file}")
            ct_path = os.path.join(ct_folder_path, ct_file)
            seg_path = os.path.join(seg_folder_path, seg_file)
            cl_path = os.path.join(cl_folder_path, cl_file)
            dm_path = os.path.join(dm_folder_path, dm_file)

            new_ct, new_seg, new_cl, new_dm = self.process(ct_path, seg_path, cl_path, dm_path, classes=self.classes)
            if new_ct != None and new_seg !=None and new_cl != None and new_dm != None:
                sitk.WriteImage(new_ct, os.path.join(self.fixed_path, 'ct', ct_file))
                sitk.WriteImage(new_seg, os.path.join(self.fixed_path, 'seg', ct_file))
                sitk.WriteImage(new_cl, os.path.join(self.fixed_path, 'cl', ct_file))
                sitk.WriteImage(new_dm, os.path.join(self.fixed_path, 'dm', ct_file))

    def process(self, ct_path, seg_path, cl_path, dm_path, classes=None):
        ct = sitk.ReadImage(ct_path, sitk.sitkFloat64)
        ct_array = sitk.GetArrayFromImage(ct)

        seg = sitk.ReadImage(seg_path, sitk.sitkInt8)
        seg_array = sitk.GetArrayFromImage(seg)

        cl = sitk.ReadImage(cl_path, sitk.sitkInt8)
        cl_array = sitk.GetArrayFromImage(cl)

        dm = sitk.ReadImage(dm_path, sitk.sitkFloat64)
        dm_array = sitk.GetArrayFromImage(dm)



        print("Ori shape:", ct_array.shape, seg_array.shape, cl_array.shape, dm_array.shape)

        if classes==2:
            # 将金标准中肝脏和肝肿瘤的标签融合为一个
            seg_array[seg_array > 0] = 1
        # # 将灰度值在阈值之外的截断掉
        # ct_array[ct_array > self.upper] = self.upper
        # ct_array[ct_array < self.lower] = self.lower
        #
        # filter_array[filter_array > self.upper] = self.upper
        # filter_array[filter_array < self.lower] = self.lower

        # 降采样，（对x和y轴进行降采样，slice轴的spacing归一化到slice_down_scale）
        ct_array = ndimage.zoom(ct_array,
                                (ct.GetSpacing()[-1] / self.slice_down_scale, self.xy_down_scale, self.xy_down_scale),
                                order=3)
        seg_array = ndimage.zoom(seg_array,
                                    (ct.GetSpacing()[-1] / self.slice_down_scale, self.xy_down_scale,
                                     self.xy_down_scale),
                                    order=0)
        cl_array = ndimage.zoom(cl_array,
                                 (ct.GetSpacing()[-1] / self.slice_down_scale, self.xy_down_scale,
                                  self.xy_down_scale),
                                 order=0)
        dm_array = ndimage.zoom(dm_array,
                                 (ct.GetSpacing()[-1] / self.slice_down_scale, self.xy_down_scale,
                                  self.xy_down_scale),
                                 order=0)

        # 找到肝脏区域开始和结束的slice，并各向外扩张
        # pdb.set_trace()
        z = np.any(seg_array, axis=(1, 2))
        start_slice, end_slice = np.where(z)[0][[0, -1]]

        # 两个方向上各扩张个slice
        if start_slice - self.expand_slice < 0:
            start_slice = 0
        else:
            start_slice -= self.expand_slice

        if end_slice + self.expand_slice >= seg_array.shape[0]:
            end_slice = seg_array.shape[0] - 1
        else:
            end_slice += self.expand_slice

        print("Cut out range:", str(start_slice) + '--' + str(end_slice))
        # 如果这时候剩下的slice数量不足size，直接放弃，这样的数据很少
        if end_slice - start_slice + 1 < self.size:
            print('Too little slice，give up the sample:', ct_path)
            return None, None, None, None
        #截取保留区域
        ct_array = ct_array[start_slice:end_slice + 1, :, :]
        seg_array = seg_array[start_slice:end_slice + 1, :, :]
        cl_array = cl_array[start_slice:end_slice + 1, :, :]
        dm_array = dm_array[start_slice:end_slice + 1, :, :]

        # slice_data = ct_array[40, :, :]
        # slice_target_branch1 = mask_array[40, :, :]
        #
        # # 使用 matplotlib 查看这个切片
        # import matplotlib.pyplot as plt
        #
        # plt.imshow(slice_data, cmap='gray')
        # plt.show()
        # plt.imshow(slice_target_branch1, cmap='gray')
        # plt.show()

        print("Preprocessed shape:", ct_array.shape, seg_array.shape, cl_array.shape, dm_array.shape)

        # 保存为对应的格式
        new_ct = sitk.GetImageFromArray(ct_array)
        new_ct.SetDirection(ct.GetDirection())
        new_ct.SetOrigin(ct.GetOrigin())
        new_ct.SetSpacing((ct.GetSpacing()[0] * int(1 / self.xy_down_scale),
                           ct.GetSpacing()[1] * int(1 / self.xy_down_scale), self.slice_down_scale))


        new_seg = sitk.GetImageFromArray(seg_array)
        new_seg.SetDirection(ct.GetDirection())
        new_seg.SetOrigin(ct.GetOrigin())
        new_seg.SetSpacing((ct.GetSpacing()[0] * int(1 / self.xy_down_scale),
                             ct.GetSpacing()[1] * int(1 / self.xy_down_scale), self.slice_down_scale))

        new_cl = sitk.GetImageFromArray(cl_array)
        new_cl.SetDirection(ct.GetDirection())
        new_cl.SetOrigin(ct.GetOrigin())
        new_cl.SetSpacing((ct.GetSpacing()[0] * int(1 / self.xy_down_scale),
                            ct.GetSpacing()[1] * int(1 / self.xy_down_scale), self.slice_down_scale))

        new_dm = sitk.GetImageFromArray(dm_array)
        new_dm.SetDirection(ct.GetDirection())
        new_dm.SetOrigin(ct.GetOrigin())
        new_dm.SetSpacing((ct.GetSpacing()[0] * int(1 / self.xy_down_scale),
                            ct.GetSpacing()[1] * int(1 / self.xy_down_scale), self.slice_down_scale))
        return new_ct, new_seg, new_cl, new_dm

    def write_train_val_name_list(self):
        ct_name_list = natsort.natsorted(os.listdir(join(self.fixed_path, "ct")))
        # seg_name_list = os.listdir(join(self.fixed_path, "seg"))
        # cl_name_list = os.listdir(join(self.fixed_path, "cl"))
        # dm_name_list = os.listdir(join(self.fixed_path, "dm"))
        data_num = len(ct_name_list)
        print('the fixed dataset total numbers of samples is :', data_num)
        random.shuffle(ct_name_list)

        assert self.valid_rate < 1.0
        train_name_list = ct_name_list[0:int(data_num * (1 - self.valid_rate))]
        val_name_list = ct_name_list[
                        int(data_num * (1 - self.valid_rate)):int(data_num * ((1 - self.valid_rate) + self.valid_rate))]

        self.write_name_list(train_name_list, "train_path_list.txt")
        self.write_name_list(val_name_list, "val_path_list.txt")

    def write_name_list(self, name_list, file_name):
        f = open(join(self.fixed_path, file_name), 'w')
        for name in name_list:
            ct_path = os.path.join(self.fixed_path, 'ct', name)
            seg_path = os.path.join(self.fixed_path, 'seg', name)
            cl_path = os.path.join(self.fixed_path, 'cl', name)
            dm_path = os.path.join(self.fixed_path, 'dm', name)
            f.write(ct_path + ' ' + seg_path + ' ' + cl_path + ' ' + dm_path + ' ' + "\n")
        f.close()

def rename():
    # 获取当前目录下所有符合 label_10_word_*.nii.gz 模式的文件
    file_paths = glob.glob('/home/dalhxwlyjsuo/guest_lizg/data/nii_data/fixed_data/labels/label_10_word_*.nii.gz')
    # 遍历所有匹配的文件
    for file_path in file_paths:
        # 提取文件名
        file_name = os.path.basename(file_path)

        # 构建新的文件名
        # new_file_name = file_name.replace('label_10_word_', 'word_')
        new_file_name = file_name.replace('..', '.')

        # 获取新的文件路径
        new_file_path = os.path.join(os.path.dirname(file_path), new_file_name)

        # 重命名文件
        os.rename(file_path, new_file_path)
        print(f'Renamed: {file_path} to {new_file_path}')


if __name__ == '__main__':
    raw_dataset_path = '/home/dalhxwlyjsuo/guest_lizg/data/nii_data/WORD/WORD-V0.1.0/imagesTr'
    label_dataset_path = '/home/dalhxwlyjsuo/guest_lizg/data/nii_data/WORD/word2colon/labelTr_colon'
    cl_dataset_path = '/home/dalhxwlyjsuo/guest_lizg/data/nii_data/cnterline/WORD/colon_skeleton_expansion'
    dm_dataset_path = '/home/dalhxwlyjsuo/guest_lizg/data/nii_data/WORD/label_bone_colon_dm'
    fixed_dataset_path = '/home/dalhxwlyjsuo/guest_lizg/data/nii_data/fixed_data'
    args = config.args
    tool = LITS_preprocess(raw_dataset_path, fixed_dataset_path, args)
    # tool.fix_data(raw_dataset_path, label_dataset_path, cl_dataset_path, dm_dataset_path)  # 对原始图像进行修剪并保存
    tool.write_train_val_name_list()  # 创建索引txt文件
    # rename()

