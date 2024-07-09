import numpy as np
import nibabel as nib
from scipy.ndimage import distance_transform_edt
import matplotlib.pyplot as plt

#计算结肠中每个像素到骨头的距离
# 加载CT图像和标签图像
ct_image_path = '/opt/data/private/DATA/WORD/WORD-V0.1.0/imagesTr/word_0002.nii.gz'
colon_image_path = '/opt/data/private/DATA/WORD/word2colon/labelTr_colon/label_10_word_0002.nii.gz'
bone_image_path = '/opt/data/private/DATA/WORD/word_bone/word_0002_label.nii.gz'

ct_image = nib.load(ct_image_path).get_fdata()
colon_image = nib.load(colon_image_path).get_fdata()
bone_image = nib.load(bone_image_path).get_fdata()

# 假设标签为：
colon_label = 1
bone_label = 2

# 创建一个空的标签图像，大小与原始图像相同
combined_label_image = np.zeros_like(ct_image, dtype=np.int32)

# 将结肠的掩膜添加到标签图像中
combined_label_image[colon_image > 0] = colon_label

# 将骨骼的掩膜添加到标签图像中
combined_label_image[bone_image > 0] = bone_label

# 保存合并后的标签图像
combined_label_image_nii = nib.Nifti1Image(combined_label_image, affine=nib.load(colon_image_path).affine)
nib.save(combined_label_image_nii, '/opt/data/private/DATA/WORD/label_bone_colon/combined_label.nii.gz')

# 重新加载合并后的标签图像
label_image = combined_label_image

# 提取结肠和骨骼掩膜
colon_mask = (label_image == colon_label)
bone_mask = (label_image == bone_label)

# 计算距离场
colon_distance_field = distance_transform_edt(~colon_mask)
bone_distance_field = distance_transform_edt(~bone_mask)

# 构建距离图：对每个结肠体素，找到其到骨骼表面的距离
distance_map = np.zeros_like(ct_image)
distance_map[colon_mask] = bone_distance_field[colon_mask]

# 可视化距离图的一个切片
slice_idx = ct_image.shape[2] // 2
plt.figure(figsize=(10, 8))
plt.imshow(distance_map[:, :, slice_idx], cmap='jet')
plt.colorbar()
plt.title('Distance Map (Colon to Bone)')
plt.show()

# # 将距离图作为网络输入的一部分
# network_input = np.stack((ct_image, distance_map), axis=-1)
#
# print("网络输入形状:", network_input.shape)
