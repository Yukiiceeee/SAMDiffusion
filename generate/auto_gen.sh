#!/bin/bash

CLASSES=("aeroplane" "bicycle" "bird" "boat" "bottle" 
         "bus" "car" "cat" "chair" "cow" 
         "diningtable" "dog" "horse" "motorbike" "person" 
         "pottedplant" "sheep" "sofa" "train" "tvmonitor")

for CLASS in "${CLASSES[@]}"; do
    echo "Processing class: $CLASS" | tee -a /data/mxy/SAMDiffusion/experiments/ablation_noise_gen.log
    
    CUDA_VISIBLE_DEVICES=4,5,6,7 python3 /data/mxy/SAMDiffusion/Stable_Diffusion/parallel_generate_VOC_Attention_AnyClass.py \
    --classes $CLASS \
    --thread_num 4 \
    --output /data/mxy/sam_data \
    --image_number 4000 \
    --MY_TOKEN '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' \
    >> /data/mxy/SAMDiffusion/experiments/ablation_noise_gen.log 2>&1

    echo "Completed processing class: $CLASS" | tee -a /data/mxy/SAMDiffusion/experiments/ablation_noise_gen.log
done

echo "All classes processed." | tee -a /data/mxy/SAMDiffusion/experiments/ablation_noise_gen.log