import os
import re

# 指定要删除文件的目录路径
directory_path = '/home/zhuyifan/Cyan_A40/mmsegmentation/data/sam_data/SegmentationClass'

# 获取指定目录下的所有文件
files = os.listdir(directory_path)

# 正则表达式匹配以数字开头的 PNG 文件
pattern = re.compile(r'^\d.*\.png$')

# 遍历所有文件并删除符合条件的文件
for file_name in files:
    # 检查文件名是否符合以数字开头且以 .png 结尾
    if pattern.match(file_name):
        # 构建完整的文件路径
        file_path = os.path.join(directory_path, file_name)
        # 删除文件
        os.remove(file_path)
        print(f"Deleted: {file_path}")
