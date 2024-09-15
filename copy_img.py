import os
import shutil

def copy_images_to_target(root_folder, output_base_path):
    # 遍历根文件夹中的所有子文件夹
    for subdir in os.listdir(root_folder):
        subdir_path = os.path.join(root_folder, subdir)
        # 检查是否为目标前缀的文件夹
        if os.path.isdir(subdir_path) and subdir.startswith('VOC_Multi_Attention'):
            # 定位到 'top_images' 文件夹
            images_folder = os.path.join(subdir_path, 'top_images')
            if os.path.exists(images_folder):
                # 遍历所有的图像文件
                for image_file in os.listdir(images_folder):
                    image_path = os.path.join(images_folder, image_file)
                    # 目标路径：将图像复制到目标目录
                    output_path = os.path.join(output_base_path, image_file)
                    os.makedirs(output_base_path, exist_ok=True)
                    shutil.copy2(image_path, output_path)  # 使用 copy2 保留元数据
                    print(f'Copied {image_path} to {output_path}')

# 指定的根文件夹路径和输出路径
root_folder = '/home/zhuyifan/Cyan_A40/sam_data'
output_base_path = '/home/zhuyifan/Cyan_A40/mmsegmentation/data/sam_data/JPEGImages'

# 执行复制操作
copy_images_to_target(root_folder, output_base_path)
