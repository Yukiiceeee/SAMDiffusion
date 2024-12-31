import numpy as np
import cv2
import os
import argparse


palette = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
           [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
           [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
           [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
           [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
           [0, 64, 128]]


category_combinations = {
    'aeroplane': {'boat', 'train', 'bus', 'car'},
    'bicycle': {'bird', 'bus', 'car', 'cat', 'cow', 'dog', 'horse', 'motorbike', 'person', 'sheep', 'train'},
    'bird': {'aeroplane', 'bicycle', 'boat', 'bus', 'car', 'cat', 'cow', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep'},
    'boat': {'aeroplane', 'bird', 'cat', 'cow', 'dog', 'person', 'train'},
    'bottle': {'bird', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'person', 'pottedplant', 'sofa', 'tvmonitor'},
    'bus': {'aeroplane', 'bicycle', 'bird', 'boat', 'car', 'cat', 'cow', 'dog', 'horse', 'motorbike', 'person', 'train'},
    'car': {'bicycle', 'bird', 'boat', 'bus', 'cat', 'dog', 'motorbike', 'person', 'train'},
    'cat': {'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'tvmonitor'},
    'chair': {'bottle', 'cat', 'dog', 'person', 'pottedplant', 'sofa', 'tvmonitor'},
    'cow': {'bird', 'bus', 'car', 'cat', 'dog', 'horse', 'person', 'sheep'},
    'dingingtable': {'bottle', 'cat', 'dog', 'person', 'pottedplant', 'sofa'},
    'dog': {'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'chair', 'cow', 'diningtable', 'cat', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'tvmonitor'},
    'horse': {'bird', 'bus', 'car', 'cat', 'dog', 'cow', 'person', 'sheep'},
    'motorbike': {'bird', 'bus', 'car', 'cat', 'cow', 'dog', 'horse', 'bicycle', 'person', 'sheep', 'train'},
    'person': {'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'dingingtable', 'dog', 'horse', 'motorbike', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'},
    'pottedplant': {'bird', 'bottle', 'cat', 'chair', 'dog', 'person', 'sheep', 'sofa', 'tvmonitor'},
    'sheep': {'bird', 'bus', 'car', 'cat', 'dog', 'horse', 'person', 'cow', 'pottedplant'},
    'sofa': {'bottle', 'cat', 'dog', 'person', 'pottedplant', 'chair', 'dingingtable', 'tvmonitor'},
    'train': {'aeroplane', 'bird', 'bus', 'car', 'boat'},
    'tvmonitor': {'bottle', 'cat', 'dog', 'chair', 'sofa', 'person', 'pottedplant'}
}


voc_class_to_index = {
    'aeroplane': 1, 'bicycle': 2, 'bird': 3, 'boat': 4, 'bottle': 5,
    'bus': 6, 'car': 7, 'cat': 8, 'chair': 9, 'cow': 10, 'diningtable': 11,
    'dog': 12, 'horse': 13, 'motorbike': 14, 'person': 15, 'pottedplant': 16,
    'sheep': 17, 'sofa': 18, 'train': 19, 'tvmonitor': 20
}


color_to_class_map = {tuple(color): idx for idx, color in enumerate(palette)}

def color_mask_to_label(mask_color, color_to_class_map):
    height, width = mask_color.shape[:2]
    label_mask = np.zeros((height, width), dtype=np.int32)
    
    for color, class_idx in color_to_class_map.items():
        color_mask = np.all(mask_color == color, axis=-1)
        label_mask[color_mask] = class_idx
    
    return label_mask

def calculate_iou(mask1, mask2):
    classes = np.unique(np.concatenate((mask1, mask2)))
    classes = classes[classes != 0]
    ious = []
    for cls in classes:
        mask1_cls = (mask1 == cls)
        mask2_cls = (mask2 == cls)
        intersection = np.logical_and(mask1_cls, mask2_cls).sum()
        union = np.logical_or(mask1_cls, mask2_cls).sum()
        iou = intersection / union if union != 0 else 0
        ious.append(iou)
    return np.mean(ious) if ious else 0

def process_masks(ground_truth_dir, noise_label_dir, color_to_class_map, top_n_per_combination=10):
    iou_dict = {}

    for gt_filename in os.listdir(ground_truth_dir):
        if gt_filename.endswith('.jpg') or gt_filename.endswith('.png'):
            gt_path = os.path.join(ground_truth_dir, gt_filename)
            
            nl_filename = gt_filename
            nl_path = os.path.join(noise_label_dir, nl_filename)
            
            if not os.path.exists(nl_path):
                print(f"Warning: noise label file not found {nl_filename}")
                continue
            
            mask_truth_color = cv2.imread(gt_path)
            mask_noise_color = cv2.imread(nl_path)
            if mask_truth_color is None or mask_noise_color is None:
                print(f"Warning: load masks failed {gt_filename}")
                continue

            
            mask_truth_color = cv2.cvtColor(mask_truth_color, cv2.COLOR_BGR2RGB)
            mask_noise_color = cv2.cvtColor(mask_noise_color, cv2.COLOR_BGR2RGB)
            mask_truth = color_mask_to_label(mask_truth_color, color_to_class_map)
            mask_noise = color_mask_to_label(mask_noise_color, color_to_class_map)

            
            category_name = nl_filename.split('_')[1]
            current_class_index = voc_class_to_index.get(category_name)

           
            unique_truth_classes = np.unique(mask_truth)
            unique_noise_classes = np.unique(mask_noise)
            non_background_truth_classes = unique_truth_classes[unique_truth_classes != 0]
            non_background_noise_classes = unique_noise_classes[unique_noise_classes != 0]
            
            if set(non_background_truth_classes) != set(non_background_noise_classes) or len(non_background_truth_classes) != len(non_background_noise_classes) or len(non_background_noise_classes) < 2:
                continue

            
            combination_list = category_combinations.get(category_name, set())
            combination_indices = {voc_class_to_index[comb_cls] for comb_cls in combination_list if comb_cls in voc_class_to_index}
            combination_indices.add(current_class_index)

            
            for cls in non_background_noise_classes:
                
                
                if cls not in combination_indices:
                    continue
                
                
                iou = calculate_iou(mask_truth, mask_noise)
                print(iou)
                if cls not in iou_dict:
                    iou_dict[cls] = []
                iou_dict[cls].append((nl_filename, iou))

    
    top_iou_dict = {}
    for cls, iou_list in iou_dict.items():
        iou_list_sorted = sorted(iou_list, key=lambda x: x[1], reverse=True)[:top_n_per_combination]
        top_iou_dict[cls] = iou_list_sorted

    return top_iou_dict

def get_top_images(iou_dict):
    top_images = []
    for cls, iou_list in iou_dict.items():
        top_images.extend([item[0] for item in iou_list])
    return top_images

def save_top_images(top_images, true_img_dir, noise_label_dir, output_image_dir, output_mask_dir):
    if not os.path.exists(output_image_dir):
        os.makedirs(output_image_dir)
    if not os.path.exists(output_mask_dir):
        os.makedirs(output_mask_dir)
    
    for filename in top_images:
        original_filename = filename.replace('.png', '.jpg')
        img_path = os.path.join(true_img_dir, original_filename)
        nl_path = os.path.join(noise_label_dir, filename)
        
        output_image_path = os.path.join(output_image_dir, original_filename)
        output_mask_path = os.path.join(output_mask_dir, filename)
        
        image = cv2.imread(img_path)
        mask = cv2.imread(nl_path)
        
        if image is not None:
            cv2.imwrite(output_image_path, image)
        if mask is not None:
            cv2.imwrite(output_mask_path, mask)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="load masks and calculate iou")
    parser.add_argument('--ground_truth_dir', type=str, required=True)
    parser.add_argument('--noise_label_dir', type=str, required=True)
    parser.add_argument('--output_image_dir', type=str, required=True)
    parser.add_argument('--output_mask_dir', type=str, required=True)
    parser.add_argument('--true_img_dir', type=str, required=True)
    parser.add_argument('--top_n', type=int, default=10)

    args = parser.parse_args()
    iou_dict = process_masks(args.ground_truth_dir, args.noise_label_dir, color_to_class_map, top_n_per_combination=args.top_n)
    top_images = get_top_images(iou_dict)
    save_top_images(top_images, args.true_img_dir, args.noise_label_dir, args.output_image_dir, args.output_mask_dir)
    
    print(f"select {len(top_images)} images,include images: {top_images}")