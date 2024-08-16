# generating Synthetic data and saving Attention Map
# python3 ./Stable_Diffusion/parallel_generate_VOC_Attention_AnyClass.py --classes bird --thread_num 8 --output ./DiffMask_VOC/ --image_number 15000

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 /home/xiangchao/home/muxinyu/SAMDiffusion/Stable_Diffusion/parallel_generate_VOC_Attention_AnyClass.py --classes cat --thread_num 8 --output ./DiffMask_VOC/ --image_number 250 --MY_TOKEN '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'
