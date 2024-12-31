from argparse import ArgumentParser
from pathlib import Path
import numpy as np
import cv2

from mmengine.model import revert_sync_batchnorm
import sys
sys.path.append('/d2/mxy/Dataset-Diffusion/mmsegmentation')
from mmseg.apis import inference_model, init_model


category_colors = {
    0: [0, 0, 0],        
    1: [128, 0, 0],      
    2: [0, 128, 0],      
    3: [128, 128, 0],    
    4: [0, 0, 128],      
    5: [128, 0, 128],    
    6: [0, 128, 128],    
    7: [128, 128, 128],  
    8: [64, 0, 0],       
    9: [192, 0, 0],      
    10: [64, 128, 0],    
    11: [192, 128, 0],   
    12: [64, 0, 128],    
    13: [192, 0, 128],   
    14: [64, 128, 128],  
    15: [192, 128, 128], 
    16: [0, 64, 0],      
    17: [128, 64, 0],    
    18: [0, 192, 0],     
    19: [128, 192, 0],   
    20: [0, 64, 128],   
}

def label_to_color_image(label):
    
    if label.ndim != 2:
        raise ValueError('Expect 2-D input label')

    colormap = np.zeros((label.shape[0], label.shape[1], 3), dtype=np.uint8)
    for label_value, color in category_colors.items():
        colormap[label == label_value] = color
    return colormap

def save_mask(mask, out_file):
    color_mask = label_to_color_image(mask)
    
    color_mask_bgr = cv2.cvtColor(color_mask, cv2.COLOR_RGB2BGR)
    cv2.imwrite(out_file, color_mask_bgr)

def main():
    parser = ArgumentParser()
    parser.add_argument('--img_dir', help='Image directory')
    parser.add_argument('--config', help='Config file')
    parser.add_argument('--checkpoint', help='Checkpoint file')
    parser.add_argument('--out-dir', default='./output', help='Path to output directory')
    parser.add_argument('--device', default='cuda:0', help='Device used for inference')
    args = parser.parse_args()

    
    model = init_model(args.config, args.checkpoint, device=args.device)
    if args.device == 'cpu':
        model = revert_sync_batchnorm(model)

    
    if args.out_dir:
        Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    
    img_dir = Path(args.img_dir)
    for img_path in img_dir.glob('*.jpg'):
       
        result = inference_model(model, str(img_path))
        seg_mask = result.pred_sem_seg.data[0].cpu().numpy().astype(np.uint8)

        
        unique_labels = np.unique(seg_mask)
        print(f'Image: {img_path.name}, Unique labels in mask: {unique_labels}')
        print(f'Image: {img_path.name}, Mask stats: min={seg_mask.min()}, max={seg_mask.max()}, mean={seg_mask.mean()}')

       
        if args.out_dir:
            out_file = Path(args.out_dir) / (img_path.stem + '.png')
        else:
            out_file = None

        
        if out_file:
            save_mask(seg_mask, str(out_file))

if __name__ == '__main__':
    main()
