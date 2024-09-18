import numpy as np
import cv2
import os

def calculate_iou(mask1, mask2):
    """
    Calculate the IoU of two masks.
        
    Parameters:
    Mask1: The first mask is a numpy array with the shape of (H, W), where H and W are the height and width of the image, respectively.
    Mask2: The second mask is a numpy array with the shape of (H, W).
        
    return:
    IoU: The IoU value of two masks.
    """
    intersection = np.logical_and(mask1, mask2)
    intersection_area = np.sum(intersection)
    
    union = np.logical_or(mask1, mask2)
    union_area = np.sum(union)
    
    iou = intersection_area / union_area if union_area != 0 else 0
    print(iou)
    return iou

def process_masks(ground_truth_dir, noise_label_dir):
    """
    Process mask images in two directories, calculate the IoU of each pair of masks, and generate a dictionary to store image names and IoU values.
        
    Parameters:
    Ground_truth.dir: The directory where the real mask image is stored.
    Noise_label-dir: A directory for storing noise mask images.
        
    return:
    Iou_ict: A dictionary containing image names and corresponding IoU values.
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
    Obtain the top N image names based on the descending order of IoU values.
        
    Parameters:
    Iou_ict: A dictionary containing image names and corresponding IoU values.
    Top_n: The number of images that need to be obtained.
        
    return:
    Top_images: An array containing the names of the top N ranked images.
    """
    sorted_iou = sorted(iou_dict.items(), key=lambda item: item[1], reverse=True)
    top_images = [item[0] for item in sorted_iou[:top_n]]
    return top_images

def save_top_images(top_images, ground_truth_dir, noise_label_dir, output_image_dir, output_mask_dir):
    """
    Save the top N images and their corresponding masks to the specified directory.
        
    Parameters:
    Top_images: An array containing the names of the top N ranked images.
    Ground_truth.dir: The directory where the real mask image is stored.
    Noise_label-dir: A directory for storing noise mask images.
    Output_image_dir: Save the output directory of the image.
    Output_mask-dir: Save the output directory of the mask.
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
    ground_truth_dir = "/home/zhuyifan/Cyan_A40/sam_data/VOC_Multi_Attention_tvmonitor_sub_4000_NoClipRetrieval_sample/ground_truth"
    noise_label_dir = "/home/zhuyifan/Cyan_A40/sam_data/VOC_Multi_Attention_tvmonitor_sub_4000_NoClipRetrieval_sample/repair_final_output"
    output_image_dir = "/home/zhuyifan/Cyan_A40/sam_data/VOC_Multi_Attention_tvmonitor_sub_4000_NoClipRetrieval_sample/top_images"
    output_mask_dir = "/home/zhuyifan/Cyan_A40/sam_data/VOC_Multi_Attention_tvmonitor_sub_4000_NoClipRetrieval_sample/top_masks"
    true_img = "/home/zhuyifan/Cyan_A40/sam_data/VOC_Multi_Attention_tvmonitor_sub_4000_NoClipRetrieval_sample/train_image"
    iou_dict = process_masks(ground_truth_dir, noise_label_dir)
    top_images = get_top_images(iou_dict, top_n=2000)
    
    save_top_images(top_images, true_img, noise_label_dir, output_image_dir, output_mask_dir)
    
    print(f"Top 2000 images based on IoU: {top_images}")