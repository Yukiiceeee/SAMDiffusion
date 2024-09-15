import os
import cv2

# 定义源文件夹路径
source_folder = "/home/zhuyifan/Cyan_A40/mmsegmentation/data/sam_data/SegmentationClass"  # 请替换为您的实际路径
output_folder = "/home/zhuyifan/Cyan_A40/mmsegmentation/data/sam_data/SegmentationClassr"  # 如果需要保存到其他文件夹，请修改为目标文件夹路径

# 确保输出目录存在
os.makedirs(output_folder, exist_ok=True)

# 遍历源文件夹中的所有文件
for filename in os.listdir(source_folder):
    # 构建完整的文件路径
    file_path = os.path.join(source_folder, filename)
    
    # 检查文件是否为图像
    if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')):
        # 读取图像
        image = cv2.imread(file_path)
        
        # 检查图像是否为3通道或更多通道
        if image is not None and len(image.shape) == 3:
            # 将图像转换为灰度图
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # 构建保存路径
            output_path = os.path.join(output_folder, filename)
            # 保存为灰度图
            cv2.imwrite(output_path, gray_image)
            # print(f"已转换并保存: {output_path}")

print("所有图像已转换为单通道灰度图。")
