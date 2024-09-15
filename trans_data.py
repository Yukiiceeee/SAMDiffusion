from PIL import Image
import os

# 指定要转换图像的目录路径
directory_path = '/home/zhuyifan/Cyan_A40/mmsegmentation/data/sam_data/JPEGImages'

# 获取目录下的所有文件
files = os.listdir(directory_path)

# 遍历目录下的所有文件
for file_name in files:
    # 检查文件是否为 PNG 文件
    if file_name.lower().endswith('.png'):
        # 构建完整的文件路径
        png_path = os.path.join(directory_path, file_name)
        
        # 打开 PNG 文件
        with Image.open(png_path) as img:
            # 设置输出的 JPG 文件路径
            jpg_path = os.path.join(directory_path, file_name[:-4] + '.jpg')
            
            # 将图像转换为 RGB 模式（PNG 可能包含透明度）
            rgb_image = img.convert('RGB')
            
            # 保存为 JPG 格式
            rgb_image.save(jpg_path, 'JPEG')
            # print(f"Converted: {png_path} to {jpg_path}")
