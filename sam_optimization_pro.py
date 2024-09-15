import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import numpy as np
import torch
import cv2
import sys
import matplotlib.pyplot as plt
from PIL import Image
from scipy.ndimage import maximum_filter
# sys.path.append(os.path.abspath('/home/zhuyifan/Cyan_A40/sam2/sam2'))
# print(sys.path) 
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor



if torch.cuda.is_available():
    device = torch.device("cuda")
if device.type == "cuda":
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
np.random.seed(3)

def show_mask(mask, ax, random_color=False, borders = True):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask = mask.astype(np.uint8)
    mask_image =  mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    if borders:
        import cv2
        contours, _ = cv2.findContours(mask,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
        # Try to smooth contours
        contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
        mask_image = cv2.drawContours(mask_image, contours, -1, (1, 1, 1, 0.5), thickness=2) 
    ax.imshow(mask_image)

def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))    

def show_masks(image, masks, scores, point_coords=None, box_coords=None, input_labels=None, borders=False):
    for i, (mask, score) in enumerate(zip(masks, scores)):
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        show_mask(mask, plt.gca(), borders=borders)
        if point_coords is not None:
            assert input_labels is not None
            show_points(point_coords, input_labels, plt.gca())
        if box_coords is not None:
            # boxes
            show_box(box_coords, plt.gca())
        if len(scores) > 1:
            plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
        plt.axis('on')
        plt.show() 

def local_maxima_filter(data, size=5):
    # 应用局部最大值滤波器
    neighborhood = maximum_filter(data, size=size)
    # 仅保留局部最大值
    maxima = (data == neighborhood)
    # 排除接近 0 的点（可选）
    threshold = np.max(data) * 0.1
    maxima[data < threshold] = False
    return maxima


sam2_checkpoint = "/home/zhuyifan/Cyan_A40/sam2/sam2_hiera_large.pt"
model_cfg = "sam2_hiera_l.yaml"

sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device)

predictor = SAM2ImagePredictor(sam2_model)

image_dir = '/home/zhuyifan/Cyan_A40/sam_data/VOC_Multi_Attention_aeroplane_sub_100_NoClipRetrieval_sample/train_image'
npy_dir = '/home/zhuyifan/Cyan_A40/sam_data/VOC_Multi_Attention_aeroplane_sub_100_NoClipRetrieval_sample/npy'
output_dir = '/home/zhuyifan/Cyan_A40/sam_data/VOC_Multi_Attention_aeroplane_sub_100_NoClipRetrieval_sample/sam_output'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.jpg')])
npy_files = sorted([f for f in os.listdir(npy_dir) if f.endswith('.npy')])

for image_file, npy_file in zip(image_files, npy_files):
    image_path = os.path.join(image_dir, image_file)
    npy_path = os.path.join(npy_dir, npy_file)
    
    image = Image.open(image_path)
    image = np.array(image.convert("RGB"))
    
    predictor.set_image(image)
    image_embedding = predictor.get_image_embedding().cpu().numpy()
    
    data = np.load(npy_path, allow_pickle=True).item()
    
    for key, value in data.items():
        if not np.all(value == 0):
            extracted_data = value
            break
    
    normalized_data = (extracted_data - np.min(extracted_data)) / (np.max(extracted_data) - np.min(extracted_data)) * 255
    normalized_data = normalized_data.astype(np.uint8)

    # 使用局部最大值滤波器提取点
    local_max = local_maxima_filter(normalized_data, size=5)
    points = np.argwhere(local_max)
    
    # 根据局部最大值的值进行排序
    values = normalized_data[points[:, 0], points[:, 1]]
    sorted_indices = np.argsort(values)[::-1]
    sorted_points = points[sorted_indices]
    
    # 将 (y, x) 坐标调整为 (x, y)
    sorted_points = sorted_points[:, ::-1]
    
    # 选取正点提示
    positive_points = sorted_points[::20][:5]
    
    # 选取负点提示
    negative_points = sorted_points[::-20][:5]
    
    input_point = np.concatenate([positive_points, negative_points])
    input_label = np.array([1]*len(positive_points) + [0]*len(negative_points))
    
    masks, scores, logits = predictor.predict(
    point_coords=input_point,
    point_labels=input_label,
    multimask_output=True,
    )
    sorted_ind = np.argsort(scores)[::-1]
    masks = masks[sorted_ind]
    scores = scores[sorted_ind]
    logits = logits[sorted_ind]

    mask_input = logits[np.argmax(scores), :, :]  # Choose the model's best mask

    masks, scores, _ = predictor.predict(
    point_coords=input_point,
    point_labels=input_label,
    mask_input=mask_input[None, :, :],
    multimask_output=False,
    )
    
    # 确保 masks 是一个二维数组
    masks = masks[0][0]
    print(masks)
    # 创建一个全黑的背景图像
    mask_image = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
    
    # 将掩码应用到背景图像上
    color = np.array([30, 144, 255], dtype=np.uint8)
    mask_image[masks] = color
    
    output_file = os.path.splitext(image_file)[0] + '_mask.png'
    output_path = os.path.join(output_dir, output_file)
    cv2.imwrite(output_path, cv2.cvtColor(mask_image, cv2.COLOR_RGB2BGR))
    
    max_value_point = sorted_points[0]
    print(f"Maximum value point coordinates: {max_value_point} {image_file}")

    # 可视化图像和点提示
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(image)
    show_points(input_point, input_label, ax)
    plt.title(f'Positive and Negative Points for {image_file}')
    plt.axis('off')
    plt.show()
