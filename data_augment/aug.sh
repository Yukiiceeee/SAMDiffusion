#!/bin/bash

BASE_INPUT="/home/zhuyifan/Cyan_A40/sam_data/VOC_Multi_Attention"
BASE_OUTPUT="/home/zhuyifan/Cyan_A40/sam_data/VOC_Multi_Attention"

CATEGORIES=("aeroplane" "bicycle" "bird" "boat" "bottle" "bus" "car" "cat" "chair" "cow" 
            "diningtable" "dog" "horse" "motorbike" "person" "pottedplant" "sheep" "sofa" 
            "train" "tvmonitor")

for CATEGORY in "${CATEGORIES[@]}"
do
    # Construct the input and output paths
    INPUT_ROOT="${BASE_INPUT}_${CATEGORY}_sub_4000_NoClipRetrieval_sample"
    OUTPUT_ROOT="${INPUT_ROOT}/aug_images"

    # Run the Python augmentation script
    echo "Processing category: ${CATEGORY}"
    python3 Augmentation_Gaussian_VOC.py --n_image 100 --input_root "$INPUT_ROOT" --out_root "$OUTPUT_ROOT"
    python3 Augmentation_occlusion_VOC.py --n_image 500 --input_root "$INPUT_ROOT" --out_root "$OUTPUT_ROOT"
    python3 Augmentation_Transformation_VOC.py --n_image 50 --input_root "$INPUT_ROOT" --out_root "$OUTPUT_ROOT"
    python3 Augmentation_Distortion_VOC.py --n_image 50 --input_root "$INPUT_ROOT" --out_root "$OUTPUT_ROOT"
    # Splicing default generate 300 images
    python3 Augmentation_splicing_VOC.py --input_root "$INPUT_ROOT" --out_root "$OUTPUT_ROOT"
done

echo "All categories processed."