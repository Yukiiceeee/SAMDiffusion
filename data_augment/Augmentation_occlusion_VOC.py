import os
import cv2
import numpy as np
from tqdm import tqdm
from random import choice, randint
import argparse
import multiprocessing as mp

def extract_foreground(image, mask):
    
    foreground = np.zeros_like(image, dtype=np.uint8)
    foreground[(mask >= 1) & (mask <= 255)] = image[(mask >= 1) & (mask <= 255)]
    return foreground, mask

def overlay_image(background, foreground, mask, top_left):
    
    x, y = top_left
    h, w = foreground.shape[:2]
    roi = background[y:y+h, x:x+w]
    
    cv2.copyTo(foreground, mask, roi)
    return background

def overlay_mask(background_mask, mask_foreground, top_left):
   
    x, y = top_left
    h, w = mask_foreground.shape[:2]

    
    roi_mask = background_mask[y:y+h, x:x+w]

    
    roi_mask[(mask_foreground >= 1) & (mask_foreground <= 255)] = mask_foreground[(mask_foreground >= 1) & (mask_foreground <= 255)]

   
    background_mask[y:y+h, x:x+w] = roi_mask

    return background_mask

def occlusion_with_mask(root, out_root, image_list, n_image=100, start_id=0, min_scale=0.3, max_scale=0.7):
    print("-------------- start augmentation: Occlusion with Mask -------------")

    for idx in tqdm(range(n_image)):
        
        image_1 = choice(image_list)
        mask_1 = image_1.replace(".jpg", ".png")

        image_2 = choice(image_list)
        mask_2 = image_2.replace(".jpg", ".png")

        
        if not os.path.exists(f"{root}/top_images/{image_1}") or not os.path.exists(f"{root}/top_images/{image_2}"):
            continue
        if not os.path.exists(f"{root}/top_masks/{mask_1}") or not os.path.exists(f"{root}/top_masks/{mask_2}"):
            continue

        
        img1 = cv2.imread(f"{root}/top_images/{image_1}")
        img2 = cv2.imread(f"{root}/top_images/{image_2}")
        mas1 = cv2.imread(f"{root}/top_masks/{mask_1}", cv2.IMREAD_GRAYSCALE)
        mas2 = cv2.imread(f"{root}/top_masks/{mask_2}", cv2.IMREAD_GRAYSCALE)

        
        mas2 = np.where((mas2 >= 1) & (mas2 <= 255), mas2, 0).astype(np.uint8)

        
        scale = np.random.uniform(min_scale, max_scale)
        img2_resized = cv2.resize(img2, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
        mas2_resized = cv2.resize(mas2, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
        mas2_resized = np.where((mas2_resized >= 1) & (mas2_resized <= 255), mas2_resized, 0).astype(np.uint8)  # 确保掩码依然是二值的

        
        foreground, mask_foreground = extract_foreground(img2_resized, mas2_resized)

       
        h2, w2 = foreground.shape[:2]
        h1, w1 = img1.shape[:2]

        
        if h2 > h1 or w2 > w1:
            continue

        
        top_left_x = randint(0, w1 - w2)
        top_left_y = randint(0, h1 - h2)

        
        combined_image = overlay_image(img1, foreground, mask_foreground, (top_left_x, top_left_y))

        
        combined_mask = overlay_mask(mas1, mask_foreground, (top_left_x, top_left_y))

        
        cv2.imwrite(f"{out_root}/Image_4/occlusion_{start_id}.jpg", combined_image)
        cv2.imwrite(f"{out_root}/Mask_4/occlusion_{start_id}.png", combined_mask)

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
        default="/home/zhuyifan/Cyan_A40/sam_data/data_aug_occlusion",
        help="the seed (for reproducible sampling)",
    )
    parser.add_argument(
        "--thread_num",
        type=int,
        default=1,
        help="the number of threads",
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

    
    manager = mp.Manager()
    result_dict = manager.dict()

    
    context = mp.get_context("spawn")  
    processes = []

    print('Start Generation')
    for i in range(opt.thread_num):
        p = context.Process(target=occlusion_with_mask, args=(opt.input_root, opt.out_root, image_list, opt.n_image, i * opt.n_image))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    result_dict = dict(result_dict)

if __name__ == "__main__":
    main()