import os
import cv2
import numpy as np
from tqdm import tqdm
import argparse
from random import choice, uniform
import multiprocessing as mp

def affine_transform(image, mask, max_rotation=30, max_scale=0.2, max_shift=20):
    
    h, w = image.shape[:2]

    
    angle = uniform(-max_rotation, max_rotation)
    scale = uniform(1 - max_scale, 1 + max_scale)
    tx = uniform(-max_shift, max_shift)
    ty = uniform(-max_shift, max_shift)

   
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, scale)
    M[0, 2] += tx 
    M[1, 2] += ty  

   
    transformed_image = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    transformed_mask = cv2.warpAffine(mask, M, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    
    transformed_mask = np.where((transformed_mask >= 181) & (transformed_mask <= 200), transformed_mask, 0).astype(np.uint8)

    return transformed_image, transformed_mask

def augment_images(root, out_root, image_list, n_image=100, start_id=0):
   
    print("-------------- start augmentation: Affine Transform -------------")

    for idx in tqdm(range(n_image)):
        
        image_name = choice(image_list)
        mask_name = image_name.replace(".jpg", ".png")

        
        if not os.path.exists(f"{root}/top_images/{image_name}") or not os.path.exists(f"{root}/top_masks/{mask_name}"):
            continue

        
        img = cv2.imread(f"{root}/top_images/{image_name}")
        mask = cv2.imread(f"{root}/top_masks/{mask_name}", cv2.IMREAD_GRAYSCALE)

        
        transformed_img, transformed_mask = affine_transform(img, mask)

        
        cv2.imwrite(f"{out_root}/Image_4/affine_{start_id}.jpg", transformed_img)
        cv2.imwrite(f"{out_root}/Mask_4/affine_{start_id}.png", transformed_mask)

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
        default="/home/zhuyifan/Cyan_A40/sam_data/data_aug_transform",
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
        p = context.Process(target=augment_images, args=(
            opt.input_root, opt.out_root, image_list, opt.n_image, i * opt.n_image))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

if __name__ == "__main__":
    main()