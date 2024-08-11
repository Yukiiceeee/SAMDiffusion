import numpy as np
import os
import argparse
from PIL import Image
import cv2
import multiprocessing
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_labels, create_pairwise_bilateral, create_pairwise_gaussian
import matplotlib.pyplot as plt

def get_findContours(mask):
    idxx = np.unique(mask)
    if len(idxx) == 1:
        return mask
    idxx = idxx[1]
    mask_instance = (mask > 0.5).astype(np.uint8)
    contours, _ = cv2.findContours(mask_instance.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    image_h, image_w = mask.shape[:2]
    gt_kernel = np.zeros((image_h, image_w), dtype='uint8')
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w * h < 1500:
            continue
        cv2.fillPoly(gt_kernel, [cnt], int(idxx))
    return gt_kernel

def _crf_inference(img, labels, t=10, n_labels=21, gt_prob=0.5):
    h, w = img.shape[:2]
    d = dcrf.DenseCRF2D(w, h, n_labels)
    U = unary_from_labels(labels, n_labels, gt_prob=gt_prob, zero_unsure=False)
    d.setUnaryEnergy(U)
    feats = create_pairwise_gaussian(sdims=(3, 3), shape=img.shape[:2])
    d.addPairwiseEnergy(feats, compat=3, kernel=dcrf.DIAG_KERNEL, normalization=dcrf.NORMALIZE_SYMMETRIC)
    feats = create_pairwise_bilateral(sdims=(80, 80), schan=(13, 13, 13), img=img, chdim=2)
    d.addPairwiseEnergy(feats, compat=10, kernel=dcrf.DIAG_KERNEL, normalization=dcrf.NORMALIZE_SYMMETRIC)
    Q = d.inference(t)
    return np.array(Q).reshape((n_labels, h, w))

def _infer_crf_with_alpha(start, step, alpha):
    for idx in range(start, len(name_list), step):
        name = name_list[idx].split("/")[-1].replace(".jpg", "")
        initial_mask_path = os.path.join(args.initial_mask_dir, f'{name}_mask.png')
        # print(initial_mask_path)
        if not os.path.isfile(initial_mask_path) or os.path.isfile(os.path.join(args.out_crf, f'{name}.png')):
            continue
        # 将图像转换为RGB模式（如果需要）
        image = Image.open(initial_mask_path)
        image = image.convert('RGB')
        image_array = np.array(image)
        # 获取图像的宽度和高度

        labels = np.mean(image_array, axis=2).astype(np.uint8)
        # print(gray_image_array)
        # width, height = image.size
        # for y in range(height):
        #     for x in range(width):
        #         print(gray_image_array[y][x])
        # initial_mask = np.array(Image.open(initial_mask_path).convert("L"))
        # print(initial_mask)
        h, w = labels.shape

        img = np.array(Image.open(os.path.join(args.infer_list, f'{name}.jpg')).convert("RGB"))

        # # Convert initial mask to label format
        # labels = initial_mask.astype(np.int32)
        labels[labels == 143] = 7  # Adjust this line based on your dataset's maximum label value
        for y in range(h):
            for x in range(w):
                print("({},{}):{}".format(y,x,labels[y][x]))
        # Debug: Print initial mask and labels
        

        crf_array = _crf_inference(img, labels)
        class_id = int(coco_category_to_id_v1[name.split("_")[1]]) + 1
        crf_array_class = crf_array[class_id]
        crf_mask = (crf_array_class > 0.5).astype(np.uint8)

        # Debug: Print CRF result
        

        res = get_findContours(crf_mask * class_id)

        # Debug: Print final mask
        

        cv2.imwrite(os.path.join(args.out_crf, f'{name}.png'), res)

        # Visualize intermediate results
        # plt.figure(figsize=(15, 5))
        # plt.subplot(1, 3, 1)
        # plt.title("Initial Mask")
        # plt.imshow(initial_mask, cmap='gray')
        # plt.subplot(1, 3, 2)
        # plt.title("CRF Result")
        # plt.imshow(crf_array_class, cmap='gray')
        # plt.subplot(1, 3, 3)
        # plt.title("Final Mask")
        # plt.imshow(res, cmap='gray')
        # plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--infer_list", default="./voc12/train_aug.txt", type=str)
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--voc12_root", default='/VOC2012', type=str)
    parser.add_argument("--initial_mask_dir", default=None, type=str)
    parser.add_argument("--out_crf", default=None, type=str)
    parser.add_argument("--dataset", default=None, type=str)
    parser.add_argument("--crf_iters", default=10, type=float)
    parser.add_argument("--alpha", default=4, type=float)

    args = parser.parse_args()
    assert args.initial_mask_dir is not None

    if args.out_crf and not os.path.exists(args.out_crf):
        os.makedirs(args.out_crf)

    name_list = [i for i in os.listdir(args.infer_list) if i.endswith('.jpg')]

    if args.dataset == "voc":
        coco_category_to_id_v1 = {
            'aeroplane': 0, 'bicycle': 1, 'bird': 2, 'boat': 3, 'bottle': 4,
            'bus': 5, 'car': 6, 'cat': 7, 'chair': 8, 'cow': 9,
            'diningtable': 10, 'dog': 11, 'horse': 12, 'motorbike': 13, 'person': 14,
            'pottedplant': 15, 'sheep': 16, 'sofa': 17, 'train': 18, 'tvmonitor': 19
        }
    elif args.dataset == "cityscapes":
        coco_category_to_id_v1 = {
            'road': 0, 'sidewalk': 1, 'building': 2, 'wall': 3, 'fence': 4,
            'pole': 5, 'traffic light': 6, 'traffic sign': 7, 'vegetation': 8, 'terrain': 9,
            'sky': 10, 'person': 11, 'rider': 12, 'car': 13, 'truck': 14,
            'bus': 15, 'train': 16, 'motorcycle': 17, 'bicycle': 18
        }

    alpha_list = ["la"]
    for alpha in alpha_list:
        p_list = []
        for i in range(args.num_workers):
            p = multiprocessing.Process(target=_infer_crf_with_alpha, args=(i, args.num_workers, alpha))
            p.start()
            p_list.append(p)
        for p in p_list:
            p.join()
        print(f'Info: Alpha {alpha} done!')