import os
import cv2
import numpy as np
from tqdm import tqdm
import random  # 确保这一行在代码的顶部
from random import choice, randint
import argparse
import multiprocessing as mp

def fisheye_effect(image, mask):
    """
    对图像和掩码应用鱼眼效果
    """
    h, w = image.shape[:2]

    # 创建相机矩阵和失真系数
    K = np.array([[w, 0, w / 2],
                  [0, w, h / 2],
                  [0, 0, 1]])
    D = np.array([-0.3, 0.1, 0, 0])  # 失真系数

    # 畸变图像
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, (w, h), cv2.CV_16SC2)
    distorted_image = cv2.remap(image, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    distorted_mask = cv2.remap(mask, map1, map2, interpolation=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT)

    # 保持掩码的二值性
    distorted_mask = np.where(distorted_mask == 127, 127, 0).astype(np.uint8)

    return distorted_image, distorted_mask

def perspective_transform(image, mask):
    """
    对图像和掩码应用透视变换
    """
    h, w = image.shape[:2]

    # 随机生成透视变换的四个点
    src_pts = np.float32([
        [randint(0, w // 4), randint(0, h // 4)],
        [randint(3 * w // 4, w), randint(0, h // 4)],
        [randint(0, w // 4), randint(3 * h // 4, h)],
        [randint(3 * w // 4, w), randint(3 * h // 4, h)]
    ])
    dst_pts = np.float32([
        [0, 0],
        [w, 0],
        [0, h],
        [w, h]
    ])

    # 计算透视变换矩阵
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    distorted_image = cv2.warpPerspective(image, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    distorted_mask = cv2.warpPerspective(mask, M, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT)

    # 保持掩码的二值性
    distorted_mask = np.where(distorted_mask == 127, 127, 0).astype(np.uint8)

    return distorted_image, distorted_mask

def augment_with_optical_distortion(root, out_root, image_list, n_image=100, start_id=0):
    print("-------------- start augmentation: Optical Distortion -------------")

    for idx in tqdm(range(n_image)):
        image_name = choice(image_list)
        mask_name = image_name.replace("jpg", "png")

        # 检查文件是否存在
        if not os.path.exists(f"{root}/top_images/{image_name}") or not os.path.exists(f"{root}/top_masks/{mask_name}"):
            continue

        # 读取图像和掩码
        img = cv2.imread(f"{root}/top_images/{image_name}")
        mas = cv2.imread(f"{root}/top_masks/{mask_name}", cv2.IMREAD_GRAYSCALE)

        # 随机选择变换类型
        if random.random() > 0.5:
            augmented_image, augmented_mask = fisheye_effect(img, mas)
        else:
            augmented_image, augmented_mask = perspective_transform(img, mas)

        # 保存结果
        cv2.imwrite(f"{out_root}/Image/optical_distortion_{start_id}.jpg", augmented_image)
        cv2.imwrite(f"{out_root}/Mask/optical_distortion_{start_id}.png", augmented_mask)

        start_id += 1

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input_root",
        type=str,
        nargs="?",
        default="/home/zhuyifan/Cyan_A40/sam_data/VOC_Multi_Attention_aeroplane_sub_4000_NoClipRetrieval_sample",
        help="config for training"
    )
    parser.add_argument(
        "--out_root",
        type=str,
        default="/home/zhuyifan/Cyan_A40/sam_data/data_aug_distortion",
        help="the seed (for reproducible sampling)",
    )
    parser.add_argument(
        "--thread_num",
        type=int,
        default=1,
        help="the number of threads",
    )
    opt = parser.parse_args()

    os.makedirs(f"{opt.out_root}/Image/", exist_ok=True)
    os.makedirs(f"{opt.out_root}/Mask/", exist_ok=True)

    image_list = os.listdir(f'{opt.input_root}/top_images')

    # 使用multiprocessing模块创建Manager
    manager = mp.Manager()
    result_dict = manager.dict()

    # 使用多进程进行数据增强
    context = mp.get_context("spawn")  # 使用spawn方法创建子进程
    processes = []

    print('Start Generation')
    for i in range(opt.thread_num):
        p = context.Process(target=augment_with_optical_distortion, args=(opt.input_root, opt.out_root, image_list, 100, i * 100))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    result_dict = dict(result_dict)

if __name__ == "__main__":
    main()