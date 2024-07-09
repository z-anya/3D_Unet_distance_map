# import os
# import numpy as np
# import SimpleITK as sitk
# from skimage import filters
#
# def segment_bones_ct_otsu(input_folder, output_folder):
#     # 创建输出文件夹
#     if not os.path.exists(output_folder):
#         os.makedirs(output_folder)
#
#     # 获取输入文件夹中所有的 NIfTI 文件
#     ct_files = [f for f in os.listdir(input_folder) if f.endswith('.nii.gz')]
#
#     # 遍历每个文件并应用 Otsu 阈值化
#     for ct_file in ct_files:
#         input_path = os.path.join(input_folder, ct_file)
#         output_path = os.path.join(output_folder, f"segmented_{ct_file}")
#
#         # 读取 NIfTI 文件
#         ct_image = sitk.ReadImage(input_path)
#         ct_array = sitk.GetArrayFromImage(ct_image)
#
#         # 对每张切片应用 Otsu 阈值化
#         segmented_slices = []
#         for slice_idx in range(ct_array.shape[0]):
#             slice_image = ct_array[slice_idx, :, :]
#             # 使用 Otsu 阈值化
#             threshold_value = filters.threshold_otsu(slice_image)
#             segmented_slice = (slice_image > threshold_value).astype(np.uint8) * 255
#             segmented_slices.append(segmented_slice)
#
#         # 将分割后的切片重建成三维数组
#         segmented_ct_array = np.array(segmented_slices)
#
#         # 保存分割后的 NIfTI 文件
#         segmented_ct_image = sitk.GetImageFromArray(segmented_ct_array)
#         segmented_ct_image.CopyInformation(ct_image)
#         sitk.WriteImage(segmented_ct_image, output_path)
#
# if __name__ == "__main__":
#     input_folder = "/opt/data/private/WORD/WORD-V0.1.0/imagesTr"
#     output_folder = "/opt/data/private/3DUNet-origin-2/3DUNet-Pytorch-master/dataset/bone"
#
#     segment_bones_ct_otsu(input_folder, output_folder)
import os
import numpy as np
import SimpleITK as sitk
from skimage import filters

def segment_bones_ct_fixed_threshold(input_folder, output_folder, fixed_threshold):
    # 创建输出文件夹
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 获取输入文件夹中所有的 NIfTI 文件
    ct_files = [f for f in os.listdir(input_folder) if f.endswith('.nii.gz')]

    # 遍历每个文件并应用固定值阈值化
    for ct_file in ct_files:
        input_path = os.path.join(input_folder, ct_file)
        output_path = os.path.join(output_folder, f"segmented_{ct_file}")

        # 读取 NIfTI 文件
        ct_image = sitk.ReadImage(input_path)
        ct_array = sitk.GetArrayFromImage(ct_image)

        # 对每张切片应用固定值阈值化
        segmented_slices = []
        for slice_idx in range(ct_array.shape[0]):
            slice_image = ct_array[slice_idx, :, :]
            # 使用固定值阈值
            threshold_value = fixed_threshold
            segmented_slice = (slice_image > threshold_value).astype(np.uint8) * 255
            segmented_slices.append(segmented_slice)

        # 将分割后的切片重建成三维数组
        segmented_ct_array = np.array(segmented_slices)

        # 保存分割后的 NIfTI 文件
        segmented_ct_image = sitk.GetImageFromArray(segmented_ct_array)
        segmented_ct_image.CopyInformation(ct_image)
        sitk.WriteImage(segmented_ct_image, output_path)

if __name__ == "__main__":
    input_folder = "/opt/data/private/WORD/WORD-V0.1.0/imagesTr"
    output_folder = "/opt/data/private/WORD/word2colon/bone_150"
    fixed_threshold = 150  # 替换为你希望的固定阈值

    segment_bones_ct_fixed_threshold(input_folder, output_folder, fixed_threshold)

