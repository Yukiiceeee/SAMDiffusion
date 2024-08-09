import numpy as np
import matplotlib.pyplot as plt

# 加载npy文件
file_path = '/data2/mxy/SAMDiffusion/DiffMask_VOC/VOC_Multi_Attention_cat_sub_1000_NoClipRetrieval_sample/npy/image_cat_200015.npy'
data = np.load(file_path,allow_pickle=True).item()
# print(data)
for key, value in data.items():
    if not np.all(value == 0):
        extracted_data = value
        break
max_value = np.max(extracted_data)
max_index = np.unravel_index(np.argmax(extracted_data), extracted_data.shape)
print(f"Maximum value: {max_value} at index: {max_index}")

# 使用matplotlib显示图像并标注最大值的点
plt.imshow(extracted_data, cmap='gray')  # 假设数据是灰度图像
plt.colorbar()
plt.scatter(max_index[1], max_index[0], color='red', marker='x')  # 用红色标注最大值的点
plt.title(f'Maximum value: {max_value} at ({max_index[1]}, {max_index[0]})')

# 保存图像到指定路径
output_path = 'path_to_save_image.png'
plt.savefig(output_path)

# 显示图像
plt.show()