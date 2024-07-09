import os
import numpy as np
import nibabel as nib

#把两个标签融合在一起
# 定义路径
ct_image_dir = '/opt/data/private/DATA/WORD/WORD-V0.1.0/imagesTr'
colon_label_dir = '/opt/data/private/DATA/WORD/word2colon/labelTr_colon'
bone_label_dir = '/opt/data/private/DATA/WORD/word_bone'
output_dir = '/opt/data/private/DATA/WORD/label_bone_colon'

# 创建输出目录（如果不存在）
os.makedirs(output_dir, exist_ok=True)

# 获取所有结肠标签文件的列表
colon_files = sorted(os.listdir(colon_label_dir))

# 假设结肠和骨骼的文件名格式匹配
for colon_file in colon_files:
    # 提取文件编号
    file_number = colon_file.split('_')[-1].split('.')[0]

    # 构建对应的骨骼标签文件名
    bone_file = f"word_{file_number}_label.nii.gz"

    # 加载NIfTI文件
    colon_path = os.path.join(colon_label_dir, colon_file)
    bone_path = os.path.join(bone_label_dir, bone_file)

    colon_img = nib.load(colon_path)
    bone_img = nib.load(bone_path)

    colon_data = colon_img.get_fdata()
    bone_data = bone_img.get_fdata()

    # 创建一个空的标签图像，大小与原始图像相同
    combined_label_image = np.zeros_like(colon_data, dtype=np.int32)

    # 假设标签为：
    colon_label = 1
    bone_label = 2

    # 将结肠的掩膜添加到标签图像中
    combined_label_image[colon_data > 0] = colon_label

    # 将骨骼的掩膜添加到标签图像中
    combined_label_image[bone_data > 0] = bone_label

    # 保存合并后的标签图像
    combined_label_image_nii = nib.Nifti1Image(combined_label_image, affine=colon_img.affine)
    output_path = os.path.join(output_dir, f"word_{file_number}.nii.gz")
    nib.save(combined_label_image_nii, output_path)

    print(f"合并后的标签图像已保存到: {output_path}")
