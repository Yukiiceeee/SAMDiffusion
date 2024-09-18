import numpy as np
from scipy import ndimage
from skimage import io, color, morphology
import matplotlib.pyplot as plt
import os
import argparse

# 设置命令行参数解析
parser = argparse.ArgumentParser(description='Process and repair images.')
parser.add_argument('--input_dir', type=str, required=True, help='Path to the input directory containing images.')
parser.add_argument('--output_dir', type=str, required=True, help='Path to the output directory to save repaired images.')
args = parser.parse_args()

input_dir = args.input_dir
output_dir = args.output_dir

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 定义类别颜色映射
category_colors = {
    'aeroplane': np.array([128, 0, 0], dtype=np.uint8),      # 深红色
    'bicycle': np.array([0, 128, 0], dtype=np.uint8),        # 深绿色
    'bird': np.array([128, 128, 0], dtype=np.uint8),         # 橄榄色
    'boat': np.array([0, 0, 128], dtype=np.uint8),           # 深蓝色
    'bottle': np.array([128, 0, 128], dtype=np.uint8),       # 紫色
    'bus': np.array([0, 128, 128], dtype=np.uint8),          # 深青色
    'car': np.array([255, 0, 0], dtype=np.uint8),            # 红色
    'cat': np.array([0, 255, 0], dtype=np.uint8),            # 绿色
    'chair': np.array([0, 0, 255], dtype=np.uint8),          # 蓝色
    'cow': np.array([255, 255, 0], dtype=np.uint8),          # 黄色
    'diningtable': np.array([255, 0, 255], dtype=np.uint8),  # 品红色
    'dog': np.array([0, 255, 255], dtype=np.uint8),          # 青色
    'horse': np.array([192, 192, 192], dtype=np.uint8),      # 银色
    'motorbike': np.array([128, 128, 128], dtype=np.uint8),  # 灰色
    'person': np.array([128, 0, 255], dtype=np.uint8),       # 紫罗兰色
    'pottedplant': np.array([255, 128, 0], dtype=np.uint8),  # 橙色
    'sheep': np.array([0, 128, 255], dtype=np.uint8),        # 天蓝色
    'sofa': np.array([128, 255, 0], dtype=np.uint8),         # 黄绿色
    'train': np.array([255, 0, 128], dtype=np.uint8),        # 玫瑰红
    'tvmonitor': np.array([0, 255, 128], dtype=np.uint8),    # 春绿色
    'background': np.array([0, 0, 0], dtype=np.uint8)        # 黑色
}

# 创建类别到标签的映射
categories = list(category_colors.keys())
label_map = {category: idx for idx, category in enumerate(categories)}
color_to_label = {}
label_to_color = {}
for category, color in category_colors.items():
    label = label_map[category]
    color_tuple = tuple(color)
    color_to_label[color_tuple] = label
    label_to_color[label] = color

background_label = label_map['background']

# 遍历输入目录中的所有图像文件
for filename in os.listdir(input_dir):
    if filename.endswith('.png'):
        # 加载图像
        image_path = os.path.join(input_dir, filename)
        image = io.imread(image_path)

        # 处理RGBA图像
        if image.shape[2] == 4:
            # 将 Alpha 通道应用到 RGB 通道
            alpha = image[:, :, 3] / 255.0
            image = image[:, :, :3] * alpha[:, :, None] + (1 - alpha[:, :, None]) * 255
            image = image.astype(np.uint8)

        # 将RGB图像转换为标签图像
        height, width, _ = image.shape
        label_image = np.zeros((height, width), dtype=np.int32)

        # 将图像展平成二维数组以加速处理
        flat_image = image.reshape(-1, 3)
        flat_labels = np.zeros((flat_image.shape[0],), dtype=np.int32)

        # 将颜色映射到标签
        for color, label in color_to_label.items():
            mask = np.all(flat_image == color, axis=1)
            flat_labels[mask] = label

        label_image = flat_labels.reshape(height, width)

        # 创建新的标签图像来存储处理后的结果
        new_label_image = np.full_like(label_image, background_label)

        # 对每个类别进行形态学操作
        for label in np.unique(label_image):
            if label == background_label:
                continue
            # 创建当前类别的二值掩码
            label_mask = label_image == label

            # 填充小孔洞
            label_mask = morphology.remove_small_holes(label_mask, area_threshold=512)

            # 去除小区域
            label_mask = morphology.remove_small_objects(label_mask, min_size=218)

            # 更新新的标签图像
            new_label_image[label_mask] = label

        # 将标签图像转换回RGB图像
        repaired_image = np.zeros((height, width, 3), dtype=np.uint8)
        for label in np.unique(new_label_image):
            color = label_to_color[label]
            mask = new_label_image == label
            repaired_image[mask] = color

        # 保存修复后的图像
        output_path = os.path.join(output_dir, filename)
        io.imsave(output_path, repaired_image)
        print(f"Repaired image saved to {output_path}")