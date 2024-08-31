from argparse import ArgumentParser
from pathlib import Path
import numpy as np
import mmcv

from mmengine.model import revert_sync_batchnorm

from mmseg.apis import inference_model, init_model
color_map = np.array([
    [0, 0, 0],        # 背景
    [128, 0, 0],      # 飞机
    [0, 128, 0],      # 自行车
    [128, 128, 0],    # 鸟
    [0, 0, 128],      # 船
    [128, 0, 128],    # 瓶子
    [0, 128, 128],    # 公车
    [128, 128, 128],  # 汽车
    [64, 0, 0],       # 猫
    [192, 0, 0],      # 椅子
    [64, 128, 0],     # 牛
    [192, 128, 0],    # 餐桌
    [64, 0, 128],     # 狗
    [192, 0, 128],    # 马
    [64, 128, 128],   # 摩托车
    [192, 128, 128],  # 人
    [0, 64, 0],       # 植物
    [128, 64, 0],     # 羊
    [0, 192, 0],      # 沙发
    [128, 192, 0],    # 火车
    [0, 64, 128],     # 显示器
], dtype=np.uint8)

def label_to_color_image(label, color_map):
    """将类别标签映射到颜色图像"""
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

    # 初始化模型
    model = init_model(args.config, args.checkpoint, device=args.device)
    if args.device == 'cpu':
        model = revert_sync_batchnorm(model)

    # 确保输出目录存在
    if args.out_dir:
        Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    # 处理所有图像
    img_dir = Path(args.img_dir)
    for img_path in img_dir.glob('*.jpg'):
        # 推理
        result = inference_model(model, str(img_path))
        seg_mask = result.pred_sem_seg.data[0].cpu().numpy()

        # 调试：打印分割结果的统计信息
        print(f'Image: {img_path.name}, Mask stats: min={seg_mask.min()}, max={seg_mask.max()}, mean={seg_mask.mean()}')

        # 确定输出文件路径
        if args.out_dir:
            out_file = Path(args.out_dir) / (img_path.stem + '_mask.png')
        else:
            out_file = None

        # 保存掩码
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