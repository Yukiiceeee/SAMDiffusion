import os
import cv2
import numpy as np
from tqdm import tqdm
from random import choice, randint
import argparse
import multiprocessing as mp

def random_occlusion(img, mask, occlusion_size_range=(20, 100)):
    """
    对图像和掩码进行随机遮挡
    :param img: 原图像
    :param mask: 原掩码
    :param occlusion_size_range: 遮挡区域的尺寸范围 (min, max)
    :return: 遮挡后的图像和掩码
    """
    h, w, _ = img.shape
    # 随机选择遮挡区域的尺寸
    occlusion_h = randint(*occlusion_size_range)
    occlusion_w = randint(*occlusion_size_range)

    # 随机选择遮挡区域的位置
    top_left_x = randint(0, w - occlusion_w)
    top_left_y = randint(0, h - occlusion_h)

    # 执行遮挡，使用黑色覆盖区域
    img[top_left_y:top_left_y + occlusion_h, top_left_x:top_left_x + occlusion_w] = [0, 0, 0]
    mask[top_left_y:top_left_y + occlusion_h, top_left_x:top_left_x + occlusion_w] = 0

    return img, mask

def augment_with_occlusion(root, out_root, image_list, n_image=100, start_id=0, occlusion_size_range=(20, 100)):
    print("-------------- start augmentation: Random Occlusion -------------")
    
    for idx in tqdm(range(n_image)):
        # 随机选择一张图像和对应的掩码
        image_name = choice(image_list)
        mask_name = image_name.replace("jpg", "png")

        # 检查文件是否存在
        if not os.path.exists(f"{root}/top_images/{image_name}") or not os.path.exists(f"{root}/top_masks/{mask_name}"):
            continue

        # 读取图像和掩码
        img = cv2.imread(f"{root}/top_images/{image_name}")
        mask = cv2.imread(f"{root}/top_masks/{mask_name}", cv2.IMREAD_GRAYSCALE)

        # 对图像和掩码进行随机遮挡
        img_occluded, mask_occluded = random_occlusion(img, mask, occlusion_size_range)

        # 保存增强后的图像和掩码
        cv2.imwrite(f"{out_root}/Image/occlusion_{start_id}.jpg", img_occluded)
        cv2.imwrite(f"{out_root}/Mask/occlusion_{start_id}.png", mask_occluded)

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
        default="/home/zhuyifan/Cyan_A40/sam_data/data_aug_erasing",
        help="output root for augmented data",
    )
    parser.add_argument(
        "--thread_num",
        type=int,
        default=1,
        help="number of threads for parallel processing",
    )
    opt = parser.parse_args()
    
    os.makedirs(f"{opt.out_root}/Image/", exist_ok=True)
    os.makedirs(f"{opt.out_root}/Mask/", exist_ok=True)
    
    image_list = os.listdir(f'{opt.input_root}/top_images')
    
    # 使用multiprocessing模块创建Manager
    manager = mp.Manager()
    result_dict = manager.dict()
    context = mp.get_context("spawn")
    processes = []

    print('Start Generation')
    for i in range(opt.thread_num):
        p = context.Process(target=augment_with_occlusion, args=(opt.input_root, opt.out_root, image_list, 100, i * 100))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    result_dict = dict(result_dict)

if __name__ == "__main__":
    main()