import numpy as np
from PIL import Image
import os
import time


def npy_to_png(npy_file, output_dir,img_cnt=200001,file_name=""):
    # 加载 .npy 文件，并允许使用 pickle
    data = np.load(npy_file, allow_pickle=True).item()
    
    # 检查数据类型和内容
    if not isinstance(data, dict):
        raise ValueError("The loaded data is not a dictionary")

    # 创建输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 处理每个键值对
    for key, image_data in data.items():
        # 检查图像数据类型和形状
        print(f"Processing key: {key}")
        print(f"Image data shape: {image_data.shape}")
        print(f"Image data dtype: {image_data.dtype}")

        # 如果 image_data 全为零，则跳过
        if np.all(image_data == 0):
            print(f"Skipping key {key} because the image data is all zeros.")
            continue

        # 如果数据不是 uint8 类型（0-255），则需要归一化
        if image_data.dtype != np.uint8:
            image_data = (255 * (image_data - np.min(image_data)) / (np.max(image_data) - np.min(image_data))).astype(np.uint8)

        # 将 numpy 数组转换为图像
        if len(image_data.shape) == 2:
            # 单通道图像
            img = Image.fromarray(image_data, mode='L')
        elif len(image_data.shape) == 3:
            # 多通道图像
            if image_data.shape[2] == 3:
                img = Image.fromarray(image_data, mode='RGB')
            elif image_data.shape[2] == 4:
                img = Image.fromarray(image_data, mode='RGBA')
            else:
                raise ValueError("Unsupported channel number in the image data")
        else:
            raise ValueError("Unsupported image shape")

        # 构造输出文件路径
        # img_cnt = img_cnt + 1
        output_file = os.path.join(output_dir, f"{file_name}.png")

        # 保存图像为 .png 文件
        img.save(output_file)
        print(f"Saved {output_file}")

def process_all_npy_files(input_dir, output_dir):
    img_cnt = 200000
    for file_name in os.listdir(input_dir):
        if file_name.endswith('.npy'):
            # print(file_name)
            # time.sleep(5)
            img_cnt = img_cnt + 1
            npy_file = os.path.join(input_dir, file_name)
            npy_to_png(npy_file, output_dir,img_cnt,file_name)

input_dir = '/data2/mxy/SAMDiffusion/DiffMask_VOC/VOC_Multi_Attention_cat_sub_1000_NoClipRetrieval_sample/npy'
output_dir = '/data2/mxy/SAMDiffusion/DiffMask_VOC/VOC_Multi_Attention_cat_sub_1000_NoClipRetrieval_sample/mask'
process_all_npy_files(input_dir, output_dir)