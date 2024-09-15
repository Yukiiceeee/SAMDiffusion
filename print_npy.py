import numpy as np
import matplotlib.pyplot as plt
import os

# 指定文件路径
file_path = '/home/zhuyifan/Cyan_A40/sam_data/VOC_Multi_Attention_cat_sub_1_NoClipRetrieval_sample/npy/image_cat_200001.npy'

# 加载 npy 文件，确保以字典形式加载
data = np.load(file_path, allow_pickle=True).item()  # 使用 .item() 将数据转换为字典
print(data)

# # 指定保存图像的路径
# save_path = '/home/zhuyifan/Cyan_A40/sam_data/multi_objects'
# os.makedirs(save_path, exist_ok=True)  # 如果文件夹不存在则创建

# # 遍历字典中的每个数组
# for key, array in data.items():
#     plt.figure(figsize=(6, 6))  # 创建一个新的图形
#     plt.imshow(array, cmap='viridis')  # 使用 colormap 'viridis' 可视化数组
#     plt.colorbar()  # 添加颜色条

#     # 构造保存图像的文件路径
#     file_name = f"array_{key}.png"
#     file_path = os.path.join(save_path, file_name)

#     # 保存图像
#     plt.savefig(file_path)
#     plt.close()  # 关闭图形以释放内存

#     print(f"Saved: {file_path}")
