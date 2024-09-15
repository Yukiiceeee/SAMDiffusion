import cv2
import os
import numpy as np

# 设置标签路径和类别数量
label_dir = '/home/zhuyifan/Cyan_A40/mmsegmentation/data/VOCdevkit/VOC2012/SegmentationClassAug'
n_classes = 21  # 假设类别数为21

def check_label_range(label_path, n_classes):
    # 读取标签图像
    label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
    
    # 获取图像中所有唯一的像素值
    unique_values = np.unique(label)
    
    # 检查是否有值超出范围 [0, n_classes-1]
    invalid_values = unique_values[(unique_values < 0) | (unique_values >= n_classes)]
    
    if len(invalid_values) > 0:
        print(f'Invalid values found in {label_path}: {invalid_values}')
        return True
    return False

def find_invalid_labels(label_dir, n_classes):
    # 遍历所有标签图像文件
    for filename in os.listdir(label_dir):
        label_path = os.path.join(label_dir, filename)
        if os.path.isfile(label_path):
            # 检查标签图像是否有超出范围的值
            if check_label_range(label_path, n_classes):
                print(f'Error in file: {label_path}')

# 执行检查
find_invalid_labels(label_dir, n_classes)
