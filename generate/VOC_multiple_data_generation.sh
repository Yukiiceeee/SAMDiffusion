#!/bin/bash

CLASSES=("aeroplane" "bicycle" "bird" "boat" "bottle" 
         "bus" "car" "cat" "chair" "cow" 
         "diningtable" "dog" "horse" "motorbike" "person" 
         "pottedplant" "sheep" "sofa" "train" "tvmonitor")

for CLASS in "${CLASSES[@]}"; do
    echo "Processing class: $CLASS"

    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 /data2/mxy/SAMDiffusion/Stable_Diffusion/parallel_generate_VOC_Multiple_Attention_AnyClass.py \
    --classes $CLASS \
    --thread_num 8 \
    --output /data2/mxy/sam_data \
    --image_number 2000 \
    --MY_TOKEN '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'

    CUDA_VISIBLE_DEVICES=4 python3 /data2/mxy/sam2/sam2_optimizition_multiple.py \
    --image_dir "/data2/mxy/sam_data/VOC_Multi_Attention_${CLASS}_sub_2000_sample_TWO/train_image" \
    --npy_dir "/data2/mxy/sam_data/VOC_Multi_Attention_${CLASS}_sub_2000_sample_TWO/npy" \
    --output_dir "/data2/mxy/sam_data/VOC_Multi_Attention_${CLASS}_sub_2000_sample_TWO/sam_output" \
    --visual_output_dir "/data2/mxy/sam_data/VOC_Multi_Attention_${CLASS}_sub_2000_sample_TWO/visual_output" \
    --attention_output_dir "/data2/mxy/sam_data/VOC_Multi_Attention_${CLASS}_sub_2000_sample_TWO/attention_output"

    rm -r /data2/mxy/sam_data/VOC_Multi_Attention_${CLASS}_sub_2000_sample_TWO/npy
    
    python3 /data2/mxy/SAMDiffusion/repair.py \
    --input_dir "/data2/mxy/sam_data/VOC_Multi_Attention_${CLASS}_sub_2000_sample_TWO/sam_output" \
    --output_dir "/data2/mxy/sam_data/VOC_Multi_Attention_${CLASS}_sub_2000_sample_TWO/repair_output"

    python /data2/mxy/SAMDiffusion/model_infer.py \
    --img_dir "/data2/mxy/sam_data/VOC_Multi_Attention_${CLASS}_sub_2000_sample_TWO/train_image" \
    --config "/data2/mxy/mmsegmentation/work_dirs/deeplabv3_r50-d8_4xb4-20k_voc12aug-512x512/deeplabv3_r50-d8_4xb4-20k_voc12aug-512x512.py" \
    --checkpoint "/data2/mxy/mmsegmentation/work_dirs/deeplabv3_r50-d8_4xb4-20k_voc12aug-512x512/iter_40000.pth" \
    --out-dir "/data2/mxy/sam_data/VOC_Multi_Attention_${CLASS}_sub_2000_sample_TWO/ground_truth" \
    --device cuda:0

    python3 /data2/mxy/SAMDiffusion/noise_learning.py \
    --ground_truth_dir "/data2/mxy/sam_data/VOC_Multi_Attention_${CLASS}_sub_2000_sample_TWO/ground_truth" \
    --noise_label_dir "/data2/mxy/sam_data/VOC_Multi_Attention_${CLASS}_sub_2000_sample_TWO/repair_output" \
    --output_image_dir "/data2/mxy/sam_data/VOC_Multi_Attention_${CLASS}_sub_2000_sample_TWO/top_images" \
    --output_mask_dir "/data2/mxy/sam_data/VOC_Multi_Attention_${CLASS}_sub_2000_sample_TWO/top_masks" \
    --true_img_dir "/data2/mxy/sam_data/VOC_Multi_Attention_${CLASS}_sub_2000_sample_TWO/train_image" \
    --top_n 50

    echo "Completed processing class: $CLASS"
done

echo "All classes processed."
