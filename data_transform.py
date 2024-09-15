import os
import shutil
import cv2
import numpy as np

# 定义路径
source_dir = "/home/zhuyifan/Cyan_A40/sam_data"  # 请替换为您的实际路径
target_image_dir = "/home/zhuyifan/Cyan_A40/mmsegmentation/data/sam_data/JPEGImages"
target_mask_dir = "/home/zhuyifan/Cyan_A40/mmsegmentation/data/sam_data/SegmentationClass"

# 创建目标目录
os.makedirs(target_image_dir, exist_ok=True)
os.makedirs(target_mask_dir, exist_ok=True)

# PASCAL VOC类别索引映射
voc_indices = {
    "aeroplane": 1, "bicycle": 2, "bird": 3, "boat": 4, "bottle": 5,
    "bus": 6, "car": 7, "cat": 8, "chair": 9, "cow": 10,
    "diningtable": 11, "dog": 12, "horse": 13, "motorbike": 14, "person": 15,
    "pottedplant": 16, "sheep": 17, "sofa": 18, "train": 19, "tvmonitor": 20
}

# 遍历所有类别文件夹
for category in os.listdir(source_dir):
    category_path = os.path.join(source_dir, category)
    if not os.path.isdir(category_path):
        continue

    # 获取类别名称并确定类别索引
    category_name = category.split('_')[3]  # 假设类别名在第四部分
    if category_name not in voc_indices:
        continue

    category_index = voc_indices[category_name]

    # 处理原始图像和掩码
    top_image_dir = os.path.join(category_path, 'top_images')
    top_mask_dir = os.path.join(category_path, 'top_masks')

    if os.path.exists(top_image_dir) and os.path.exists(top_mask_dir):
        for image_file in os.listdir(top_image_dir):
            # 转移原始图像
            src_image_path = os.path.join(top_image_dir, image_file)
            dst_image_path = os.path.join(target_image_dir, f"{category_name}_{image_file}")
            shutil.move(src_image_path, dst_image_path)

            # 处理掩码图像
            mask_file = os.path.splitext(image_file)[0] + ".png"  # 假设掩码文件是PNG格式
            src_mask_path = os.path.join(top_mask_dir, mask_file)
            if not os.path.exists(src_mask_path):
                continue

            # 读取掩码并转换为灰度图
            mask = cv2.imread(src_mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                continue

            # 将掩码前景设置为类别索引值
            mask = np.where(mask > 0, category_index, 0).astype(np.uint8)

            # 将掩码转换为3维灰度图
            gray_mask = cv2.merge([mask, mask, mask])

            # 保存处理后的灰度掩码
            dst_mask_path = os.path.join(target_mask_dir, f"{category_name}_{mask_file}")
            cv2.imwrite(dst_mask_path, gray_mask)

    # 处理增强数据
    aug_image_dir = os.path.join(category_path, 'aug_images', 'Image')
    aug_mask_dir = os.path.join(category_path, 'aug_images', 'Mask')

    if os.path.exists(aug_image_dir) and os.path.exists(aug_mask_dir):
        for image_file in os.listdir(aug_image_dir):
            # 转移增强的原始图像
            src_image_path = os.path.join(aug_image_dir, image_file)
            dst_image_path = os.path.join(target_image_dir, f"aug_{category_name}_{image_file}")
            shutil.move(src_image_path, dst_image_path)

            # 处理增强的掩码图像
            mask_file = os.path.splitext(image_file)[0] + ".png"  # 假设掩码文件是PNG格式
            src_mask_path = os.path.join(aug_mask_dir, mask_file)
            if not os.path.exists(src_mask_path):
                continue

            # 读取掩码并转换为灰度图
            mask = cv2.imread(src_mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                continue

            # 将掩码前景设置为类别索引值
            mask = np.where(mask > 0, category_index, 0).astype(np.uint8)

            # 将掩码转换为3维灰度图
            gray_mask = cv2.merge([mask, mask, mask])

            # 保存处理后的灰度掩码
            dst_mask_path = os.path.join(target_mask_dir, f"aug_{category_name}_{mask_file}")
            cv2.imwrite(dst_mask_path, gray_mask)

print("所有图像和掩码已成功处理并保存。")
