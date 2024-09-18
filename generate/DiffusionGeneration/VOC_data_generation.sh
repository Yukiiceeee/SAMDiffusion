# generating Synthetic data and saving Attention Map
# python3 ./Stable_Diffusion/parallel_generate_VOC_Attention_AnyClass.py --classes bird --thread_num 8 --output ./DiffMask_VOC/ --image_number 15000

CATEGORIES=("aeroplane" "bicycle" "bird" "boat" "bottle" "bus" "car" "cat" "chair" "cow" 
            "diningtable" "dog" "horse" "motorbike" "person" "pottedplant" "sheep" "sofa" 
            "train" "tvmonitor")

CUDA_VISIBLE_DEVICES=0 python3 /home/zhuyifan/Cyan_A40/SAMDiffusion/Stable_Diffusion/parallel_generate_VOC_Attention_AnyClass.py --classes sofa --thread_num 1 --output /home/zhuyifan/Cyan_A40/sam_data --image_number 50 --MY_TOKEN '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'