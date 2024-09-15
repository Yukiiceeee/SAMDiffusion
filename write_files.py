import os

# 定义源文件夹和目标txt文件路径
source_folder = "/home/zhuyifan/Cyan_A40/mmsegmentation/data/sam_data/JPEGImages"  # 请替换为您的实际路径
output_txt = "/home/zhuyifan/Cyan_A40/mmsegmentation/data/sam_data/ImageSets/Segmentation/train.txt"   # 目标txt文件路径

# 支持的图像文件扩展名
image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff", ".webp"}

# 打开目标txt文件进行写入
with open(output_txt, 'w') as file:
    # 遍历源文件夹中的所有文件
    for filename in os.listdir(source_folder):
        # 获取文件的完整路径
        file_path = os.path.join(source_folder, filename)
        
        # 检查是否为文件而不是文件夹，并且文件不以数字开头且扩展名为图像类型
        if os.path.isfile(file_path) and not filename[0].isdigit() and os.path.splitext(filename)[1].lower() in image_extensions:
            # 去除文件的扩展名
            name_without_extension = os.path.splitext(filename)[0]
            # 写入文件名（去除扩展名）到txt文件
            file.write(name_without_extension + '\n')

print(f"文件名已写入至 {output_txt}.")
