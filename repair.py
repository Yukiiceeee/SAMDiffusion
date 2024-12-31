import numpy as np
from scipy import ndimage
from skimage import io, color, morphology
import matplotlib.pyplot as plt
import os
import argparse


parser = argparse.ArgumentParser(description='Process and repair images.')
parser.add_argument('--input_dir', type=str, required=True, help='Path to the input directory containing images.',default='/home/zhuyifan/Cyan_A40/sam_data/VOC_Multi_Attention_cat_sub_1_sample/sam_output')
parser.add_argument('--output_dir', type=str, required=True, help='Path to the output directory to save repaired images.',default='/home/zhuyifan/Cyan_A40/sam_data/VOC_Multi_Attention_cat_sub_1_sample/repair_output')
args = parser.parse_args()

input_dir = args.input_dir
output_dir = args.output_dir

if not os.path.exists(output_dir):
    os.makedirs(output_dir)


category_colors = {
    'aeroplane': np.array([128, 0, 0], dtype=np.uint8),      
    'bicycle': np.array([0, 128, 0], dtype=np.uint8),        
    'bird': np.array([128, 128, 0], dtype=np.uint8),        
    'boat': np.array([0, 0, 128], dtype=np.uint8),           
    'bottle': np.array([128, 0, 128], dtype=np.uint8),       
    'bus': np.array([0, 128, 128], dtype=np.uint8),          
    'car': np.array([128, 128, 128], dtype=np.uint8),        
    'cat': np.array([64, 0, 0], dtype=np.uint8),             
    'chair': np.array([192, 0, 0], dtype=np.uint8),          
    'cow': np.array([64, 128, 0], dtype=np.uint8),           
    'diningtable': np.array([192, 128, 0], dtype=np.uint8),  
    'dog': np.array([64, 0, 128], dtype=np.uint8),           
    'horse': np.array([192, 0, 128], dtype=np.uint8),        
    'motorbike': np.array([64, 128, 128], dtype=np.uint8),   
    'person': np.array([192, 128, 128], dtype=np.uint8),     
    'pottedplant': np.array([0, 64, 0], dtype=np.uint8),     
    'sheep': np.array([128, 64, 0], dtype=np.uint8),         
    'sofa': np.array([0, 192, 0], dtype=np.uint8),           
    'train': np.array([128, 192, 0], dtype=np.uint8),        
    'tvmonitor': np.array([0, 64, 128], dtype=np.uint8),    
    'background': np.array([0, 0, 0], dtype=np.uint8)        
}


categories = list(category_colors.keys())
label_map = {category: idx for idx, category in enumerate(categories)}
color_to_label = {}
label_to_color = {}
for category, color in category_colors.items():
    label = label_map[category]
    color_tuple = tuple(color)
    color_to_label[color_tuple] = label
    label_to_color[label] = color

background_label = label_map['background']


for filename in os.listdir(input_dir):
    if filename.endswith('.png'):
        
        image_path = os.path.join(input_dir, filename)
        image = io.imread(image_path)

        
        if image.shape[2] == 4:
            
            alpha = image[:, :, 3] / 255.0
            image = image[:, :, :3] * alpha[:, :, None] + (1 - alpha[:, :, None]) * 255
            image = image.astype(np.uint8)

        
        height, width, _ = image.shape
        label_image = np.zeros((height, width), dtype=np.int32)

        
        flat_image = image.reshape(-1, 3)
        flat_labels = np.zeros((flat_image.shape[0],), dtype=np.int32)

        
        for color, label in color_to_label.items():
            mask = np.all(flat_image == color, axis=1)
            flat_labels[mask] = label

        label_image = flat_labels.reshape(height, width)

        
        new_label_image = np.full_like(label_image, background_label)

        
        for label in np.unique(label_image):
            if label == background_label:
                continue
            
            label_mask = label_image == label

            
            label_mask = morphology.remove_small_holes(label_mask, area_threshold=512)

           
            label_mask = morphology.remove_small_objects(label_mask, min_size=218)

           
            new_label_image[label_mask] = label

        
        repaired_image = np.zeros((height, width, 3), dtype=np.uint8)
        for label in np.unique(new_label_image):
            color = label_to_color[label]
            mask = new_label_image == label
            repaired_image[mask] = color

        
        output_path = os.path.join(output_dir, filename)
        io.imsave(output_path, repaired_image)
        print(f"Repaired image saved to {output_path}")