import os
import cv2
import numpy as np
from tqdm import tqdm
import argparse
from random import choice
import multiprocessing as mp

def gaussian_blur_images(root, out_root, image_list, n_image=100, start_id=0, kernel_size=(15, 15), sigma=5):
   
    print("-------------- start augmentation: Gaussian Blur -------------")

    for idx in tqdm(range(n_image)):
       
        image_name = choice(image_list)
        mask_name = image_name.replace(".jpg", ".png")

        
        if not os.path.exists(f"{root}/top_images/{image_name}") or not os.path.exists(f"{root}/top_masks/{mask_name}"):
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            print(mask_name)

        
        img = cv2.imread(f"{root}/top_images/{image_name}")
        mask = cv2.imread(f"{root}/top_masks/{mask_name}")

        
        blurred_img = cv2.GaussianBlur(img, kernel_size, sigma)

        
        cv2.imwrite(f"{out_root}/Image_4/blurred_{start_id}.jpg", blurred_img)
        cv2.imwrite(f"{out_root}/Mask_4/blurred_{start_id}.png", mask)

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
        default="/home/zhuyifan/Cyan_A40/sam_data/data_aug_gaussian",
        help="Output directory for augmented data",
    )
    parser.add_argument(
        "--thread_num",
        type=int,
        default=1,
        help="Number of threads",
    )
    parser.add_argument(
        "--n_image",
        type=int,
        default=100,
        help="Number of images to process",
    )
    opt = parser.parse_args()

    
    os.makedirs(f"{opt.out_root}/Image_4/", exist_ok=True)
    os.makedirs(f"{opt.out_root}/Mask_4/", exist_ok=True)

   
    image_list = os.listdir(f'{opt.input_root}/top_images')

    
    context = mp.get_context("spawn")  
    processes = []

    print('Start Generation')
    for i in range(opt.thread_num):
        p = context.Process(target=gaussian_blur_images, args=(
            opt.input_root, opt.out_root, image_list, opt.n_image, i * opt.n_image))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

if __name__ == "__main__":
    main()