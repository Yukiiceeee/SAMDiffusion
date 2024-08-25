# generating Synthetic data and saving Attention Map
# python3 ./Stable_Diffusion/parallel_generate_VOC_Attention_AnyClass.py --classes bird --thread_num 8 --output ./DiffMask_VOC/ --image_number 15000

CUDA_VISIBLE_DEVICES=0 python3 /home/zhuyifan/Cyan_A40/SAMDiffusion/Stable_Diffusion/parallel_generate_VOC_Attention_AnyClass.py --classes cat --thread_num 1 --output ./DiffMask_VOC/ --image_number 500 --MY_TOKEN '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'
