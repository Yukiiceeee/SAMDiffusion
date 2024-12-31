import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.ndimage import maximum_filter
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
    npy_files = sorted([f for f in os.listdir(args.npy_dir) if f.endswith('.npy')])

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

    categories = list(category_colors.keys())

    
    def extract_category_from_path(path, categories):
        for part in path.split(os.sep):
            for category in categories:
                if category in part:
                    return category
        return None

    
    category = extract_category_from_path(args.image_dir, categories)
    if category is None:
        raise ValueError("Could not find a valid category in image_dir path.")

    
    mask_color = category_colors[category]

    for image_file, npy_file in zip(image_files, npy_files):
        image_path = os.path.join(args.image_dir, image_file)
        npy_path = os.path.join(args.npy_dir, npy_file)
        
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        predictor.set_image(image)
        
        data = np.load(npy_path, allow_pickle=True).item()
        extracted_data = None

        for value in data.values():
            if isinstance(value, np.ndarray) and value.ndim == 2 and np.any(value):
                extracted_data = value
                break

        if extracted_data is None:
            continue
        
        normalized_data = (extracted_data - np.min(extracted_data)) / (np.max(extracted_data) - np.min(extracted_data)) * 255
        normalized_data = normalized_data.astype(np.uint8)
        
       
        average_attention = np.mean(normalized_data)
        print(f"Average attention value for {image_file}: {average_attention}")
        
        upper_w = args.upper_w
        upper_b = args.upper_b
        low_w = args.low_w
        low_b = args.low_b
        U = upper_w * np.log(average_attention) - upper_b
        upper_threshold = np.clip(U, 100, 200)
        L = low_w * average_attention - low_b
        lower_threshold = np.clip(L, 5, 40)

        if upper_threshold <= lower_threshold:
            lower_threshold = upper_threshold - 5
            if lower_threshold < 5:
                lower_threshold = 5

        print(f"For {image_file}: upper_threshold = {upper_threshold}, lower_threshold = {lower_threshold}")

        positive_points = []
        negative_points = []

        H, W = normalized_data.shape
        window_size = args.window_size
        step_size = args.window_step
        positive_threshold_ratio = 0.5
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
                        idx = interval_indices[selected_idx]
                        pos_y = y_start + idx[0]
                        pos_x = x_start + idx[1]
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
                        idx = interval_indices[selected_idx]
                        neg_y = y_start + idx[0]
                        neg_x = x_start + idx[1]
                        negative_points.append([neg_x, neg_y])
                        
                        processed_attention_map[y_start:y_end, x_start:x_end] = np.maximum(
                            processed_attention_map[y_start:y_end, x_start:x_end],
                            window
                        )

        positive_points = np.array(positive_points)
        negative_points = np.array(negative_points)
       
        new_positive_points = np.empty((0, 2), dtype=int)

        if len(positive_points) > 0:
            cluster_k = args.k
            unique_positive_points = np.unique(positive_points, axis=0)
            n_unique = len(unique_positive_points)
            if cluster_k == 0:
                if len(unique_positive_points) < 10:
                    selected_k = len(unique_positive_points)
                else:
                    selected_k = 10
                valid_points = np.argwhere(normalized_data > upper_threshold)
                if len(valid_points) < selected_k:
                    selected_indices = np.arange(len(valid_points))
                else:
                    selected_indices = np.random.choice(len(valid_points), selected_k, replace=False)
                new_positive_points = valid_points[selected_indices][:, ::-1]
                new_positive_points = np.round(new_positive_points).astype(int)
            else:
                n_clusters = min(cluster_k, n_unique)
                modified_kmeans = ModifiedKMeans(n_clusters=n_clusters, distance_weight=0.1)
                modified_kmeans.fit(positive_points)
                new_positive_points = modified_kmeans.cluster_centers_
                new_positive_points = np.round(new_positive_points).astype(int)
       
        positive_points = np.round(new_positive_points).astype(int)
        negative_points = np.round(negative_points).astype(int)
        if len(positive_points) > 0 and len(negative_points) > 0:
            input_point = np.concatenate([positive_points, negative_points], axis=0)
            input_label = np.array([1]*len(positive_points) + [0]*len(negative_points))
        elif len(positive_points) > 0:
            input_point = positive_points
            input_label = np.array([1]*len(positive_points))
        elif len(negative_points) > 0:
            input_point = negative_points
            input_label = np.array([0]*len(negative_points))
        else:
            input_point = np.empty((0, 2), dtype=int)
            input_label = np.array([])

        masks, scores, logits = predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=True,
        )
       
        if masks.shape[0] >= 3:
            mask = masks[2].astype(bool)
        else:
            mask = masks[-1].astype(bool)
        
       
        base_name = os.path.splitext(image_file)[0]
        mask_image = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
        mask_image[mask] = mask_color
        
        save_path = os.path.join(args.output_dir, f"{base_name}_mask.png")
        plt.imsave(save_path, mask_image)
        

        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(image)
        if len(input_point) > 0:
            ax.scatter(
                input_point[input_label == 1, 0],  
                input_point[input_label == 1, 1],  
                color='green',
                marker='*',
                s=375,
                edgecolor='white',
                linewidth=1.25,
                label='Positive Points'
            )
            ax.scatter(
                input_point[input_label == 0, 0],  
                input_point[input_label == 0, 1],  
                color='red',
                marker='*',
                s=375,
                edgecolor='white',
                linewidth=1.25,
                label='Negative Points'
            )
        plt.title(f'Positive and Negative Points for {image_file}')
        plt.axis('off')
        plt.legend()
        
        visual_save_path = os.path.join(args.visual_output_dir, f"{base_name}_visual.png")
        plt.savefig(visual_save_path, bbox_inches='tight')
        plt.close(fig)

        
        fig, ax = plt.subplots(figsize=(10, 10))
        cax = ax.imshow(normalized_data, cmap=green_cmap, interpolation='nearest')
        plt.title(f'Attention Map for {image_file}')
        plt.axis('off')
        cbar = fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)  
        cbar.set_label('Attention Strength', rotation=270, labelpad=15)  
        attention_save_path = os.path.join(args.attention_output_dir, f"{base_name}_attention.png")
        plt.savefig(attention_save_path, bbox_inches='tight')
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(10, 10))
        cax = ax.imshow(processed_attention_map, cmap=green_cmap, interpolation='nearest')
        plt.title(f'Processed Attention Map for {image_file}')
        plt.axis('off')
        cbar = fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Attention Strength', rotation=270, labelpad=15)
        processed_attention_save_path = os.path.join(args.attention_output_dir, f"{base_name}_processed_attention.png")
        plt.savefig(processed_attention_save_path, bbox_inches='tight')
        plt.close(fig)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SAM2 Point Segmentation Script')
    parser.add_argument('--image_dir', type=str, default='/home/zhuyifan/Cyan_A40/sam_data/VOC_Multi_Attention_sofa_sub_50_sample/train_image', help='Path to the image directory')
    parser.add_argument('--npy_dir', type=str, default='/home/zhuyifan/Cyan_A40/sam_data/VOC_Multi_Attention_sofa_sub_50_sample/npy', help='Path to the npy files directory')
    parser.add_argument('--output_dir', type=str, default='/home/zhuyifan/Cyan_A40/sam_data/VOC_Multi_Attention_tvmonitor_sub_50_NoClipRetrieval_sample/sam_output', help='Path to save the segmentation masks')
    parser.add_argument('--visual_output_dir', type=str, default='/home/zhuyifan/Cyan_A40/sam_data/VOC_Multi_Attention_tvmonitor_sub_50_NoClipRetrieval_sample/visual_output', help='Path to save the visualizations')
    parser.add_argument('--attention_output_dir', type=str, default='/home/zhuyifan/Cyan_A40/sam_data/VOC_Multi_Attention_tvmonitor_sub_50_NoClipRetrieval_sample/attention_output', help='Path to save the attention maps')
    parser.add_argument('--sam_checkpoint', type=str, default='/d2/mxy/sam2/sam2_hiera_large.pt', help='Path to the SAM checkpoint file')
    parser.add_argument('--model_cfg', type=str, default='sam2_hiera_l.yaml', help='Path to the SAM model configuration file')
    parser.add_argument('--k', type=int, default=10, help='k clusters')
    parser.add_argument('--window_size', type=int, default=64, help='windows_size')
    parser.add_argument('--window_step', type=int, default=32, help='windows_step')
    parser.add_argument('--upper_w', type=float, default=40, help='upper function w')
    parser.add_argument('--upper_b', type=float, default=80, help='upper function b')
    parser.add_argument('--low_w', type=float, default=0.25, help='upper function b')
    parser.add_argument('--low_b', type=float, default=8, help='upper function b')

    args = parser.parse_args()
    main(args)