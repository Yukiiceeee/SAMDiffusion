import numpy as np
import cv2
import matplotlib.pyplot as plt
from segment_anything import sam_model_registry, SamPredictor
from segment_anything.utils.onnx import SamOnnxModel
import onnxruntime
import torch
import warnings
import os

def show_mask(mask, ax):
    color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))   

checkpoint = "/home/zhuyifan/Cyan_A40/segment-anything/sam_vit_h_4b8939.pth"
model_type = "vit_h"

sam = sam_model_registry[model_type](checkpoint=checkpoint)
onnx_model_path = "sam_onnx_example.onnx"

onnx_model = SamOnnxModel(sam, return_single_mask=True)

dynamic_axes = {
    "point_coords": {1: "num_points"},
    "point_labels": {1: "num_points"},
}

embed_dim = sam.prompt_encoder.embed_dim
embed_size = sam.prompt_encoder.image_embedding_size
mask_input_size = [4 * x for x in embed_size]
dummy_inputs = {
    "image_embeddings": torch.randn(1, embed_dim, *embed_size, dtype=torch.float),
    "point_coords": torch.randint(low=0, high=1024, size=(1, 5, 2), dtype=torch.float),
    "point_labels": torch.randint(low=0, high=4, size=(1, 5), dtype=torch.float),
    "mask_input": torch.randn(1, 1, *mask_input_size, dtype=torch.float),
    "has_mask_input": torch.tensor([1], dtype=torch.float),
    "orig_im_size": torch.tensor([1500, 2250], dtype=torch.float),
}
output_names = ["masks", "iou_predictions", "low_res_masks"]

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    with open(onnx_model_path, "wb") as f:
        torch.onnx.export(
            onnx_model,
            tuple(dummy_inputs.values()),
            f,
            export_params=True,
            verbose=False,
            opset_version=17,
            do_constant_folding=True,
            input_names=list(dummy_inputs.keys()),
            output_names=output_names,
            dynamic_axes=dynamic_axes,
        )    

ort_session = onnxruntime.InferenceSession(onnx_model_path)
sam.to(device='cuda:0')
predictor = SamPredictor(sam)

image_dir = '/home/zhuyifan/Cyan_A40/sam_data/VOC_Multi_Attention_sofa_sub_4000_NoClipRetrieval_sample/train_image'
npy_dir = '/home/zhuyifan/Cyan_A40/sam_data/VOC_Multi_Attention_sofa_sub_4000_NoClipRetrieval_sample/npy'
output_dir = '/home/zhuyifan/Cyan_A40/sam_data/VOC_Multi_Attention_sofa_sub_4000_NoClipRetrieval_sample/sam_output'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.jpg')])
npy_files = sorted([f for f in os.listdir(npy_dir) if f.endswith('.npy')])

for image_file, npy_file in zip(image_files, npy_files):
    image_path = os.path.join(image_dir, image_file)
    npy_path = os.path.join(npy_dir, npy_file)
    
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    predictor.set_image(image)
    image_embedding = predictor.get_image_embedding().cpu().numpy()
    
    data = np.load(npy_path, allow_pickle=True).item()
    
    for key, value in data.items():
        if not np.all(value == 0):
            extracted_data = value
            break
    
    normalized_data = (extracted_data - np.min(extracted_data)) / (np.max(extracted_data) - np.min(extracted_data)) * 255
    normalized_data = normalized_data.astype(np.uint8)

    points = np.argwhere(normalized_data)
    values = normalized_data[points[:, 0], points[:, 1]]
    
    sorted_indices = np.argsort(values)[::-1]
    sorted_points = points[sorted_indices]
    
    sorted_points = sorted_points[:, ::-1]
    
    # 选取正点提示
    positive_points = sorted_points[::20][:5]
    
    # 选取负点提示
    negative_points = sorted_points[::-20][:5]
    
    input_point = np.concatenate([positive_points, negative_points])
    input_label = np.array([1]*len(positive_points) + [0]*len(negative_points))
    
    mask = list(data.values())[0]
    
    onnx_mask_input = mask[:256, :256][np.newaxis, np.newaxis, :, :].astype(np.float32)
    
    onnx_coord = np.concatenate([input_point, np.array([[0.0, 0.0]])], axis=0)[None, :, :]
    onnx_label = np.concatenate([input_label, np.array([-1])], axis=0)[None, :].astype(np.float32)
    
    onnx_coord = predictor.transform.apply_coords(onnx_coord, image.shape[:2]).astype(np.float32)
    onnx_has_mask_input = np.ones(1, dtype=np.float32)
    
    ort_inputs = {
        "image_embeddings": image_embedding,
        "point_coords": onnx_coord,
        "point_labels": onnx_label,
        "mask_input": onnx_mask_input,
        "has_mask_input": onnx_has_mask_input,
        "orig_im_size": np.array(image.shape[:2], dtype=np.float32)
    }
    
    masks, _, _ = ort_session.run(None, ort_inputs)
    masks = masks > predictor.model.mask_threshold
    
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