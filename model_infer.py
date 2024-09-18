from argparse import ArgumentParser
from pathlib import Path
import numpy as np
import mmcv

from mmengine.model import revert_sync_batchnorm

from mmseg.apis import inference_model, init_model
color_map = np.array([
    [0, 0, 0],     
    [128, 0, 0],   
    [0, 128, 0],   
    [128, 128, 0], 
    [0, 0, 128],   
    [128, 0, 128], 
    [0, 128, 128], 
    [128, 128, 128],
    [64, 0, 0],    
    [192, 0, 0],   
    [64, 128, 0],  
    [192, 128, 0], 
    [64, 0, 128],  
    [192, 0, 128], 
    [64, 128, 128],
    [192, 128, 128],
    [0, 64, 0],    
    [128, 64, 0],  
    [0, 192, 0],   
    [128, 192, 0], 
    [0, 64, 128],  
], dtype=np.uint8)

def label_to_color_image(label, color_map):
    if label.ndim != 2:
        raise ValueError('Expect 2-D input label')

    colormap = np.zeros((label.shape[0], label.shape[1], 3), dtype=np.uint8)
    for i in range(len(color_map)):
        colormap[label == i, :] = color_map[i]
    return colormap

def save_mask(mask, out_file, color_map):
    color_mask = label_to_color_image(mask, color_map)
    mmcv.imwrite(color_mask, out_file)

def main():
    parser = ArgumentParser()
    parser.add_argument('img_dir', help='Image directory')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument('--out-dir', default='/home/zhuyifan/Cyan_A40/sam_data/VOC_Multi_Attention_bird_sub_4000_NoClipRetrieval_sample/ground_truth', help='Path to output directory')
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
        seg_mask = result.pred_sem_seg.data[0].cpu().numpy()

        print(f'Image: {img_path.name}, Mask stats: min={seg_mask.min()}, max={seg_mask.max()}, mean={seg_mask.mean()}')

        if args.out_dir:
            out_file = Path(args.out_dir) / (img_path.stem + '_mask.png')
        else:
            out_file = None

        if out_file:
            # print(111)
            # print(seg_mask)
            # height, width = seg_mask.shape
            # for y in range(height):
            #     for x in range(width):
            #         pixel_value = seg_mask[y, x]
            #         if pixel_value != 0:
            #             print(f'Non-zero pixel at ({x}, {y}): {pixel_value}')

            save_mask(seg_mask, str(out_file),color_map)

if __name__ == '__main__':
    main()