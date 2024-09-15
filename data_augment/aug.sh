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
    python3 Augmentation_Gaussian_VOC.py --input_root "$INPUT_ROOT" --out_root "$OUTPUT_ROOT"
done

echo "All categories processed."