import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.ndimage import label
from PIL import Image
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
import torch
import argparse
from sklearn.cluster import KMeans
from matplotlib.colors import LinearSegmentedColormap
from sklearn.base import BaseEstimator, ClusterMixin
from scipy.spatial.distance import pdist, squareform

class ModifiedKMeans(BaseEstimator, ClusterMixin):
    def __init__(self, n_clusters=10, max_iter=300, distance_weight=0.1, tol=1e-4):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.distance_weight = distance_weight
        self.tol = tol
        
    def _compute_center_distances(self):
        if len(self.cluster_centers_) <= 1:
            return 0
        distances = pdist(self.cluster_centers_)
        return np.mean(distances)
    
    def _update_centers(self, X, labels):
        new_centers = np.zeros((self.n_clusters, X.shape[1]))
        for k in range(self.n_clusters):
            if np.sum(labels == k) > 0:
                cluster_points = X[labels == k]
                base_center = np.mean(cluster_points, axis=0)
                
                other_centers = np.delete(self.cluster_centers_, k, axis=0)
                if len(other_centers) > 0:
                    direction = np.mean(other_centers - base_center, axis=0)
                    direction_norm = np.linalg.norm(direction)
                    if direction_norm > 0:
                        direction = direction / direction_norm
                        new_centers[k] = base_center - self.distance_weight * direction
                    else:
                        new_centers[k] = base_center
                else:
                    new_centers[k] = base_center
            else:
                new_centers[k] = X[np.random.randint(X.shape[0])]
        return new_centers
    
    def fit(self, X):
        rng = np.random.RandomState(42)
        idx = rng.permutation(X.shape[0])[:self.n_clusters]
        self.cluster_centers_ = X[idx].copy()
        
        for iteration in range(self.max_iter):
            old_centers = self.cluster_centers_.copy()
            
            distances = np.sqrt(((X[:, np.newaxis, :] - self.cluster_centers_[np.newaxis, :, :]) ** 2).sum(axis=2))
            labels = np.argmin(distances, axis=1)
            
            self.cluster_centers_ = self._update_centers(X, labels)

            center_shift = np.sqrt(((self.cluster_centers_ - old_centers) ** 2).sum(axis=1)).mean()
            if center_shift < self.tol:
                break
                
        self.labels_ = labels
        return self

    def predict(self, X):
        distances = np.sqrt(((X[:, np.newaxis, :] - self.cluster_centers_[np.newaxis, :, :]) ** 2).sum(axis=2))
        return np.argmin(distances, axis=1)

classes = {
    0: 'background',
    1: 'aeroplane',
    2: 'bicycle',
    3: 'bird',
    4: 'boat',
    5: 'bottle',
    6: 'bus',
    7: 'car',
    8: 'cat',
    9: 'chair',
    10: 'cow',
    11: 'diningtable',
    12: 'dog',
    13: 'horse',
    14: 'motorbike',
    15: 'person',
    16: 'pottedplant',
    17: 'sheep',
    18: 'sofa',
    19: 'train',
    20: 'tvmonitor'
}

def average_attention_by_category(attention_files, attention_dir):
    category_files = {}

    for attention_file in attention_files:
        filename_parts = attention_file.split('_')
        if len(filename_parts) >= 3:
            category = filename_parts[1]
            category = category.split('.')[0]
            
            if category not in category_files:
                category_files[category] = []
            category_files[category].append(attention_file)

    averaged_data = {}
    for category, files in category_files.items():
        attention_sum = None
        count = 0
        for file in files:
            file_path = os.path.join(attention_dir, file)
            data = np.load(file_path, allow_pickle=True)
            
            if data is not None:
                if attention_sum is None:
                    attention_sum = data
                else:
                    attention_sum += data
                count += 1
        if attention_sum is not None and count > 0:
            averaged_data[category] = attention_sum / count
    
    return averaged_data

def main(args):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    if device.type == "cuda":
        torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

    for directory in [args.output_dir, args.visual_output_dir, args.attention_output_dir]:
        if not os.path.exists(directory):
            os.makedirs(directory)

    image_files = sorted([f for f in os.listdir(args.image_dir) if f.endswith('.jpg')])
    npy_dirs = sorted([f for f in os.listdir(args.npy_dir) if os.path.isdir(os.path.join(args.npy_dir, f))])

    sam2_model = build_sam2(args.model_cfg, args.sam_checkpoint, device=device)
    predictor = SAM2ImagePredictor(sam2_model)

    green_cmap = LinearSegmentedColormap.from_list('green_cmap', ['white', 'green'])

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

    for image_file, npy_dir in zip(image_files, npy_dirs):
        image_path = os.path.join(args.image_dir, image_file)
        attention_dir = os.path.join(args.npy_dir, npy_dir)  

        attention_files = sorted([f for f in os.listdir(attention_dir) if f.endswith('.npy')])
        averaged_attention = average_attention_by_category(attention_files, attention_dir)
        
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        predictor.set_image(image)

        
        methods = ['clustering', 'top_attention', 'random_sampling']
        
        for method in methods:
            masks_list = []  
            input_points_list = []
            input_labels_list = []
            normalized_data_list = []
            processed_attention_maps_list = []
            categories_list = []

            for category, attention_data in averaged_attention.items():
                normalized_data = (attention_data - np.min(attention_data)) / (np.max(attention_data) - np.min(attention_data)) * 255
                normalized_data = normalized_data.astype(np.uint8)
                normalized_data_list.append(normalized_data)
                categories_list.append(category)
                
                average_attention = np.mean(normalized_data)
                print(f"Average attention value for {image_file}, category {category}: {average_attention}")

                U = 40 * np.log(average_attention) - 80
                upper_threshold = np.clip(U, 100, 200)

                L = 0.25 * average_attention - 8
                lower_threshold = np.clip(L, 5, 40)

                if upper_threshold <= lower_threshold:
                    lower_threshold = upper_threshold - 5
                    if lower_threshold < 5:
                        lower_threshold = 5

                print(f"For {image_file}, category {category}: upper_threshold = {upper_threshold}, lower_threshold = {lower_threshold}")

                positive_points = []
                negative_points = []

                H, W = normalized_data.shape
                window_size = 64
                step_size = 32
                positive_threshold_ratio = 0.1
                negative_threshold_ratio = 0.3
                processed_attention_map = np.zeros_like(normalized_data, dtype=np.uint8)

                for y_start in range(0, H, step_size):
                    for x_start in range(0, W, step_size):
                        y_end = min(y_start + window_size, H)
                        x_end = min(x_start + window_size, W)
                        window = normalized_data[y_start:y_end, x_start:x_end]

                        positive_indices = np.argwhere(window > upper_threshold)
                        num_positive = len(positive_indices)
                        if num_positive > 0 and num_positive / window.size >= positive_threshold_ratio:
                            num_intervals = num_positive // 1000 + 1
                            for i in range(num_intervals):
                                start_idx = i * 1000
                                end_idx = min((i + 1) * 1000, num_positive)
                                interval_indices = positive_indices[start_idx:end_idx]
                                if len(interval_indices) == 0:
                                    continue
                                selected_idx = np.random.choice(len(interval_indices))
                                idx_pt = interval_indices[selected_idx]
                                pos_y = y_start + idx_pt[0]
                                pos_x = x_start + idx_pt[1]
                                positive_points.append([pos_x, pos_y])
                                
                                processed_attention_map[y_start:y_end, x_start:x_end] = np.maximum(
                                    processed_attention_map[y_start:y_end, x_start:x_end],
                                    window
                                )

                        negative_indices = np.argwhere(window < lower_threshold)
                        num_negative = len(negative_indices)
                        if num_negative > 0 and num_negative / window.size >= negative_threshold_ratio:
                            num_intervals = num_negative // 1500 + 1
                            for i in range(num_intervals):
                                start_idx = i * 1500
                                end_idx = min((i + 1) * 1500, num_negative)
                                interval_indices = negative_indices[start_idx:end_idx]
                                if len(interval_indices) == 0:
                                    continue
                                selected_idx = np.random.choice(len(interval_indices))
                                idx_pt = interval_indices[selected_idx]
                                neg_y = y_start + idx_pt[0]
                                neg_x = x_start + idx_pt[1]
                                negative_points.append([neg_x, neg_y])
                                
                                processed_attention_map[y_start:y_end, x_start:x_end] = np.maximum(
                                    processed_attention_map[y_start:y_end, x_start:x_end],
                                    window
                                )
                processed_attention_maps_list.append(processed_attention_map)

                positive_points = np.array(positive_points)
                negative_points = np.array(negative_points)

                
                if method == 'clustering':
                    
                    if len(positive_points) > 0:
                        n_clusters = min(10, len(positive_points))
                        modified_kmeans = ModifiedKMeans(n_clusters=n_clusters, distance_weight=0.1)
                        modified_kmeans.fit(positive_points)
                        new_positive_points = modified_kmeans.cluster_centers_
                        new_positive_points = np.round(new_positive_points).astype(int)
                    else:
                        new_positive_points = np.empty((0, 2), dtype=int)

                elif method == 'top_attention':
                    
                    all_points = np.argwhere(normalized_data > 0)
                    attention_values = normalized_data[all_points[:, 0], all_points[:, 1]]
                    
                    if len(all_points) > 0:
                        top_indices = np.argsort(attention_values)[-10:]
                        new_positive_points = all_points[top_indices][:, ::-1]
                        new_positive_points = np.round(new_positive_points).astype(int)
                    else:
                        new_positive_points = np.empty((0, 2), dtype=int)

                else:  
                    
                    valid_points = np.argwhere(normalized_data > upper_threshold)
                    
                    if len(valid_points) > 10:
                        random_indices = np.random.choice(len(valid_points), 10, replace=False)
                        new_positive_points = valid_points[random_indices][:, ::-1]
                        new_positive_points = np.round(new_positive_points).astype(int)
                    elif len(valid_points) > 0:
                        new_positive_points = valid_points[:, ::-1]
                    else:
                        new_positive_points = np.empty((0, 2), dtype=int)

                if len(negative_points) > 0:
                    new_negative_points = negative_points.reshape(-1, 2)
                else:
                    new_negative_points = np.empty((0, 2), dtype=int)
                
                if len(new_positive_points) > 0 and len(new_negative_points) > 0:
                    input_point = np.concatenate([new_positive_points, new_negative_points], axis=0)
                    input_label = np.array([1]*len(new_positive_points) + [0]*len(new_negative_points))
                elif len(new_positive_points) > 0:
                    input_point = new_positive_points
                    input_label = np.array([1]*len(new_positive_points))
                elif len(new_negative_points) > 0:
                    input_point = new_negative_points
                    input_label = np.array([0]*len(new_negative_points))
                else:
                    input_point = np.empty((0, 2), dtype=int)
                    input_label = np.array([])
                
                input_points_list.append(input_point)
                input_labels_list.append(input_label)

                
                masks, scores, logits = predictor.predict(
                    point_coords=input_point,
                    point_labels=input_label,
                    multimask_output=True,
                )

                if masks.shape[0] >= 3:
                    mask = masks[2].astype(bool)
                else:
                    mask = masks[-1].astype(bool)

                masks_list.append(mask)

            if len(masks_list) == 0:
                print(f"No masks generated for {image_file}, method {method}. Skipping.")
                continue

            
            final_mask_image = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
            
            
            category_indices = [list(category_colors.keys()).index(cat) + 1 for cat in categories_list]
            class_masks = [mask.astype(np.uint8) * idx for mask, idx in zip(masks_list, category_indices)]
            
        
            combined_class_mask = np.zeros_like(class_masks[0])
            for class_mask in class_masks:
                combined_class_mask += class_mask
            
            
            overlap_mask = combined_class_mask > np.array(category_indices).max()
            labeled_overlap, num_features = label(overlap_mask)
            
            
            for i in range(1, num_features + 1):
                region_mask = labeled_overlap == i
                region_attention_averages = []
                
                for idx, (normalized_data, mask) in enumerate(zip(normalized_data_list, masks_list)):
                    category = categories_list[idx]
                    attention_values = normalized_data[region_mask]
                    mean_attention = np.mean(attention_values) if attention_values.size > 0 else 0
                    region_attention_averages.append((mean_attention, category))
                
                max_mean_attention, max_category = max(region_attention_averages, key=lambda x: x[0])
                color = category_colors.get(max_category, np.array([255, 255, 255], dtype=np.uint8))
                final_mask_image[region_mask] = color
            
            
            non_overlap_mask = np.logical_not(overlap_mask)
            for idx, mask in enumerate(masks_list):
                category = categories_list[idx]
                color = category_colors.get(category, np.array([255, 255, 255], dtype=np.uint8))
                update_mask = mask & non_overlap_mask
                final_mask_image[update_mask] = color

            
            base_name = os.path.splitext(image_file)[0]
            save_path = os.path.join(args.output_dir, f"{base_name}_mask_{method}.png")
            plt.imsave(save_path, final_mask_image)

            
            fig, ax = plt.subplots(figsize=(10, 10))
            
            plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
            
            ax.imshow(image)

            
            for idx, (input_point, input_label) in enumerate(zip(input_points_list, input_labels_list)):
                if len(input_point) > 0:
                    category = categories_list[idx]
                    color = category_colors.get(category, np.array([255, 255, 255], dtype=np.uint8)) / 255.0
                    
                    
                    pos_points = ax.scatter(
                        input_point[input_label == 1, 0],
                        input_point[input_label == 1, 1],
                        color=color,
                        marker='*',
                        s=200,
                        edgecolor='white',
                        linewidth=1.25,
                        label=f'Positive Points ({category})'
                    )
                    
                    
                    neg_points = ax.scatter(
                        input_point[input_label == 0, 0],
                        input_point[input_label == 0, 1],
                        color=color,
                        marker='o',
                        s=200,
                        edgecolor='black',
                        linewidth=1.25,
                        label=f'Negative Points ({category})'
                    )

            
            plt.axis('off')

           
            handles, labels = ax.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            legend = ax.legend(by_label.values(), by_label.keys(), 
                              loc='upper right',
                              bbox_to_anchor=(1, 1),
                              bbox_transform=ax.transAxes,
                              framealpha=0.8,  
                              edgecolor='white')  

            
            visual_save_path = os.path.join(args.visual_output_dir, f"{base_name}_visual_{method}.png")
            plt.savefig(visual_save_path, 
                        bbox_inches='tight',  
                        pad_inches=0,  
                        dpi='figure')  
            plt.close(fig)

            
            fig, ax = plt.subplots(figsize=(10, 10))
            ax.imshow(image)  
            
           
            non_black_mask = np.any(final_mask_image > 0, axis=-1)
            overlay = np.zeros_like(image, dtype=np.float32)
            overlay[non_black_mask] = final_mask_image[non_black_mask]
            
            
            ax.imshow(overlay, alpha=0.7)
            
            
            for idx, (input_point, input_label) in enumerate(zip(input_points_list, input_labels_list)):
                if len(input_point) > 0:
                    category = categories_list[idx]
                    color = category_colors.get(category, np.array([255, 255, 255])) / 255.0
                    
                    ax.scatter(
                        input_point[input_label == 1, 0],
                        input_point[input_label == 1, 1],
                        color=color,
                        marker='*',
                        s=200,
                        edgecolor='white',
                        linewidth=1.25,
                        label=f'Positive Points ({category})'
                    )
                    
                    ax.scatter(
                        input_point[input_label == 0, 0],
                        input_point[input_label == 0, 1],
                        color=color,
                        marker='o',
                        s=200,
                        edgecolor='black',
                        linewidth=1.25,
                        label=f'Negative Points ({category})'
                    )
            
            plt.title(f'Final Segmentation with Points ({method}) for {image_file}')
            plt.axis('off')
            
            handles, labels = ax.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            ax.legend(by_label.values(), by_label.keys())
            
            merged_visual_save_path = os.path.join(args.visual_output_dir, f"{base_name}_merged_visual_{method}.png")
            plt.savefig(merged_visual_save_path, bbox_inches='tight', dpi=300)
            plt.close(fig)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SAM2 Multi-Attention Segmentation Script')
    parser.add_argument('--image_dir', type=str, default='/home/zhuyifan/Cyan_A40/sam_data/VOC_Multi_Attention_bicycle_sub_50_sample_TWO/train_image', help='Path to the image directory')
    parser.add_argument('--npy_dir', type=str, default='/home/zhuyifan/Cyan_A40/sam_data/VOC_Multi_Attention_bicycle_sub_50_sample_TWO/npy', help='Path to the npy directories')
    parser.add_argument('--output_dir', type=str, default='/home/zhuyifan/Cyan_A40/sam_data/VOC_Multi_Attention_bicycle_sub_50_sample_TWO/sam_output', help='Path to save the segmentation masks')
    parser.add_argument('--visual_output_dir', type=str, default='/home/zhuyifan/Cyan_A40/sam_data/VOC_Multi_Attention_bicycle_sub_50_sample_TWO/visual_output', help='Path to save the visualizations')
    parser.add_argument('--attention_output_dir', type=str, default='/home/zhuyifan/Cyan_A40/sam_data/VOC_Multi_Attention_bicycle_sub_50_sample_TWO/attention_output', help='Path to save the attention maps')
    parser.add_argument('--sam_checkpoint', type=str, default='/data2/mxy/sam2/sam2_hiera_large.pt', help='Path to the SAM checkpoint file')
    parser.add_argument('--model_cfg', type=str, default='sam2_hiera_l.yaml', help='Path to the SAM model configuration file')

    args = parser.parse_args()
    main(args)