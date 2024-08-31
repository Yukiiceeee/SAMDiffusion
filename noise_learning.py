import numpy as np
import cv2
import os

def calculate_iou(mask1, mask2):
    """
    计算两个掩码的IoU。
    
    参数:
    mask1: 第一个掩码，形状为 (H, W) 的 numpy 数组，其中 H 和 W 分别是图像的高度和宽度。
    mask2: 第二个掩码，形状为 (H, W) 的 numpy 数组。
    
    返回:
    iou: 两个掩码的IoU值。
    """
    # 计算交集
    intersection = np.logical_and(mask1, mask2)
    intersection_area = np.sum(intersection)
    
    # 计算并集
    union = np.logical_or(mask1, mask2)
    union_area = np.sum(union)
    
    # 计算IoU
    iou = intersection_area / union_area if union_area != 0 else 0
    print(iou)
    return iou

def process_masks(ground_truth_dir, noise_label_dir):
    """
    处理两个目录中的掩码图像，计算每对掩码的IoU，并生成一个字典存储图像名称和IoU值。
    
    参数:
    ground_truth_dir: 存储真实掩码图像的目录。
    noise_label_dir: 存储噪声掩码图像的目录。
    
    返回:
    iou_dict: 包含图像名称和对应IoU值的字典。
    """
    iou_dict = {}
    
    for gt_filename in os.listdir(ground_truth_dir):
        if gt_filename.endswith('.jpg') or gt_filename.endswith('.png'):
            gt_path = os.path.join(ground_truth_dir, gt_filename)
            nl_filename = gt_filename.replace('.jpg', '_mask.png')
            # nl_filename = gt_filename.rsplit('.', 1)[0] + '_mask.png'
            nl_path = os.path.join(noise_label_dir, nl_filename)
            
            if not os.path.exists(nl_path):
                print(f"Warning: Corresponding noise label file not found for {gt_filename}")
                continue
            
            mask_truth = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
            mask_noise = cv2.imread(nl_path, cv2.IMREAD_GRAYSCALE)
            # height,width = mask_noise.shape
            # for y in range(height):  # 遍历每一行
            #     for x in range(width):  # 遍历每一列
            #         pixel_value = mask_noise[y, x]  # 访问像素值
            #         print(f"Pixel at ({y}, {x}): {pixel_value}")
            if mask_truth is None or mask_noise is None:
                print(f"Warning: Failed to read mask images for {gt_filename}")
                continue
            
            # for i in range(mask_truth.shape[0]):
            #     for j in range(mask_truth.shape[1]):
            #         value = mask_truth[i, j]
            #         print(value)
            #         if value >= 10 and value <= 30:
            #             mask_truth[i, j] = 127
            #         else:
            #             mask_truth[i, j] = 0
            mask_truth[mask_truth != 0] = 255
            mask_noise[mask_noise != 0] = 255
            
            iou = calculate_iou(mask1=mask_truth, mask2=mask_noise)
            iou_dict[gt_filename] = iou
    
    return iou_dict

def get_top_images(iou_dict, top_n=2000):
    """
    根据IoU值的降序顺序，获取排名前N的图像名称。
    
    参数:
    iou_dict: 包含图像名称和对应IoU值的字典。
    top_n: 需要获取的图像数量。
    
    返回:
    top_images: 包含排名前N的图像名称的数组。
    """
    sorted_iou = sorted(iou_dict.items(), key=lambda item: item[1], reverse=True)
    top_images = [item[0] for item in sorted_iou[:top_n]]
    return top_images

def save_top_images(top_images, ground_truth_dir, noise_label_dir, output_image_dir, output_mask_dir):
    """
    将排名前N的图像及其对应的掩码保存到指定的目录中。
    
    参数:
    top_images: 包含排名前N的图像名称的数组。
    ground_truth_dir: 存储真实掩码图像的目录。
    noise_label_dir: 存储噪声掩码图像的目录。
    output_image_dir: 保存图像的输出目录。
    output_mask_dir: 保存掩码的输出目录。
    """
    if not os.path.exists(output_image_dir):
        os.makedirs(output_image_dir)
    if not os.path.exists(output_mask_dir):
        os.makedirs(output_mask_dir)
    
    for filename in top_images:
        print(f"ground truth dir:{ground_truth_dir}")
        print(f"filename:{filename}")
        original_filename = filename.replace('_mask.png', '.jpg')
        gt_path = os.path.join(ground_truth_dir, original_filename)
        nl_filename = filename.replace('.jpg', '_mask.png')
        nl_path = os.path.join(noise_label_dir, nl_filename)
        
        output_image_path = os.path.join(output_image_dir, filename)
        output_mask_path = os.path.join(output_mask_dir, nl_filename)
        print(f"Reading image from: {gt_path}")
        image = cv2.imread(gt_path)
        print(f"Reading mask from: {nl_path}")
        mask = cv2.imread(nl_path, cv2.IMREAD_GRAYSCALE)
        
        if image is not None:
            cv2.imwrite(output_image_path, image)
        if mask is not None:
            cv2.imwrite(output_mask_path, mask)

if __name__ == "__main__":
    ground_truth_dir = "/home/zhuyifan/Cyan_A40/sam_data/VOC_Multi_Attention_bird_sub_4000_NoClipRetrieval_sample/ground_truth"
    noise_label_dir = "/home/zhuyifan/Cyan_A40/sam_data/VOC_Multi_Attention_bird_sub_4000_NoClipRetrieval_sample/repair_final_output"
    output_image_dir = "/home/zhuyifan/Cyan_A40/sam_data/VOC_Multi_Attention_bird_sub_4000_NoClipRetrieval_sample/top_images"
    output_mask_dir = "/home/zhuyifan/Cyan_A40/sam_data/VOC_Multi_Attention_bird_sub_4000_NoClipRetrieval_sample/top_masks"
    true_img = "/home/zhuyifan/Cyan_A40/sam_data/VOC_Multi_Attention_bird_sub_4000_NoClipRetrieval_sample/train_image"
    iou_dict = process_masks(ground_truth_dir, noise_label_dir)
    top_images = get_top_images(iou_dict, top_n=2000)
    
    save_top_images(top_images, true_img, noise_label_dir, output_image_dir, output_mask_dir)
    
    print(f"Top 2000 images based on IoU: {top_images}")