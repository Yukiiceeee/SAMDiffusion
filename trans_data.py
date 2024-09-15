import os

# 定义目标文件夹路径
target_folder = "/home/zhuyifan/Cyan_A40/mmsegmentation/data/sam_data/JPEGImages"  # 请替换为您的实际路径

# 遍历目标文件夹中的所有文件
for filename in os.listdir(target_folder):
    # 检查文件扩展名是否为.jpg（不区分大小写）
    if filename.lower().endswith(".jpg"):
        # 构建完整的文件路径
        file_path = os.path.join(target_folder, filename)
        
        # 删除文件
        os.remove(file_path)
        print(f"已删除: {file_path}")

print("所有以 .jpg 为后缀的文件已删除。")