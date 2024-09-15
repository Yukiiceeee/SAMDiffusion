import os
import json
import shutil
from tqdm import tqdm
import cv2
import numpy as np
import argparse
from random import choice
import random

def splicing_1x2(root, out_root, image_list, n_image=100, start_id=0):
    print("-------------- start augmentation: 1x2 -------------")

    for idx in tqdm(range(n_image)):
        image_1 = choice(image_list)
        mask_1 = image_1.replace("jpg", "png")

        image_2 = choice(image_list)
        mask_2 = image_2.replace("jpg", "png")

        if "jpg" not in image_1 or "jpg" not in image_2:
            continue

        if not os.path.exists("{}/top_masks/{}".format(root, mask_1)) or not os.path.exists("{}/top_masks/{}".format(root, mask_2)):
            continue

        img1 = cv2.imread("{}/top_images/{}".format(root, image_1))
        img2 = cv2.imread("{}/top_images/{}".format(root, image_2))

        mas1 = cv2.imread("{}/top_masks/{}".format(root, mask_1))
        mas2 = cv2.imread("{}/top_masks/{}".format(root, mask_2))

        if random.random() > 0.5:
            image = np.concatenate([img1, img2], axis=1)
            mask = np.concatenate([mas1, mas2], axis=1)
        else:
            image = np.concatenate((img1, img2))
            mask = np.concatenate((mas1, mas2))

        # Resize to 512x512
        image_resized = cv2.resize(image, (512, 512), interpolation=cv2.INTER_LINEAR)
        mask_resized = cv2.resize(mask, (512, 512), interpolation=cv2.INTER_NEAREST)

        cv2.imwrite("{}/Image/splicing_{}.jpg".format(out_root, start_id), image_resized)
        cv2.imwrite("{}/Mask/splicing_{}.png".format(out_root, start_id), mask_resized)

        start_id += 1

def splicing_NxN(root, out_root, image_list, n_image=100, start_id=0, size=2):
    print("-------------- start augmentation: {}x{} -------------".format(size, size))

    for idx in tqdm(range(n_image)):
        list_image = []
        list_mask = []
        for x in range(size):
            image_1 = choice(image_list)
            mask_1 = image_1.replace("jpg", "png")

            img1 = cv2.imread("{}/top_images/{}".format(root, image_1))
            mas1 = cv2.imread("{}/top_masks/{}".format(root, mask_1))

            for y in range(size - 1):
                image_2 = choice(image_list)
                mask_2 = image_2.replace("jpg", "png")

                img2 = cv2.imread("{}/top_images/{}".format(root, image_2))
                mas2 = cv2.imread("{}/top_masks/{}".format(root, mask_2))

                img1 = np.concatenate([img1, img2], axis=1)
                mas1 = np.concatenate([mas1, mas2], axis=1)

            list_image.append(img1)
            list_mask.append(mas1)

        list_image_ha = list_image[0]
        list_mask_ha = list_mask[0]
        for i in range(1, size):
            list_image_ha = np.concatenate((list_image_ha, list_image[i]))
            list_mask_ha = np.concatenate((list_mask_ha, list_mask[i]))

        # Resize to 512x512
        image_resized = cv2.resize(list_image_ha, (512, 512), interpolation=cv2.INTER_LINEAR)
        mask_resized = cv2.resize(list_mask_ha, (512, 512), interpolation=cv2.INTER_NEAREST)

        cv2.imwrite("{}/Image/splicing_{}.jpg".format(out_root, start_id), image_resized)
        cv2.imwrite("{}/Mask/splicing_{}.png".format(out_root, start_id), mask_resized)

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
        default="/home/zhuyifan/Cyan_A40/sam_data/data_aug_splicing3",
        help="the seed (for reproducible sampling)",
    )
    parser.add_argument(
        "--thread_num",
        type=int,
        default=3,
        help="the seed (for reproducible sampling)",
    )
    opt = parser.parse_args()

    os.makedirs("{}/Image/".format(opt.out_root), exist_ok=True)
    os.makedirs("{}/Mask/".format(opt.out_root), exist_ok=True)

    image_list = os.listdir('{}/top_images'.format(opt.input_root))

    import multiprocessing as mp
    mp = mp.get_context("spawn")
    processes = []

    print('Start Generation')
    for i in range(opt.thread_num):
        if i == 0:
            p = mp.Process(target=splicing_1x2, args=(opt.input_root, opt.out_root, image_list, 3000, 0))
            p.start()
            processes.append(p)
        elif i == 1:
            p = mp.Process(target=splicing_NxN, args=(opt.input_root, opt.out_root, image_list, 200, 15001, 2))
            p.start()
            processes.append(p)
        elif i == 2:
            p = mp.Process(target=splicing_NxN, args=(opt.input_root, opt.out_root, image_list, 100, 35001, 3))
            p.start()
            processes.append(p)
        elif i == 3:
            p = mp.Process(target=splicing_NxN, args=(opt.input_root, opt.out_root, image_list, 100, 55001, 5))
            p.start()
            processes.append(p)
        elif i == 4:
            p = mp.Process(target=splicing_NxN, args=(opt.input_root, opt.out_root, image_list, 1000, 65001, 6))
            p.start()
            processes.append(p)
        elif i == 5:
            p = mp.Process(target=splicing_NxN, args=(opt.input_root, opt.out_root, image_list, 1000, 75001, 7))
            p.start()
            processes.append(p)
        elif i == 6:
            p = mp.Process(target=splicing_NxN, args=(opt.input_root, opt.out_root, image_list, 1000, 85001, 8))
            p.start()
            processes.append(p)

    for p in processes:
        p.join()

if __name__ == "__main__":
    main()