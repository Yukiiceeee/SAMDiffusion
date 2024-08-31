import numpy as np
from scipy import ndimage
from skimage import io, color
import matplotlib.pyplot as plt
import os

# 定义输入和输出目录
input_dir = '/home/zhuyifan/Cyan_A40/sam_data/VOC_Multi_Attention_bird_sub_4000_NoClipRetrieval_sample/repair_output'
output_dir = '/home/zhuyifan/Cyan_A40/sam_data/VOC_Multi_Attention_bird_sub_4000_NoClipRetrieval_sample/repair_final_output'

# 确保输出目录存在
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 遍历输入目录中的所有图像文件
for filename in os.listdir(input_dir):
    if filename.endswith('.png'):
        # 加载图像
        image_path = os.path.join(input_dir, filename)
        image = io.imread(image_path)

        # 将图像转换为灰度图
        gray_image = color.rgb2gray(image)
        # height,width = gray_image.shape
        # for y in range(height):  # 遍历每一行
        #     for x in range(width):  # 遍历每一列
        #         pixel_value = gray_image[y, x]  # 访问像素值
        #         print(f"Pixel at ({y}, {x}): {pixel_value}")
        # 创建掩码：掩码区域为True，其他为False
        mask = gray_image > 0.4  # 黑色区域为需要修复的部分

        # 使用线性插值修复图像
        repaired_image = gray_image.copy()

        # 定义邻域的半径
        radius = 4  # 2 对应 5x5 的邻域

        # 获取掩码为True的位置
        missing_pixels = np.where(mask)

        # 对每个缺失像素，使用邻域插值
        for y, x in zip(*missing_pixels):
            # 获取邻近像素的值（忽略掩码区域）
            neighborhood = gray_image[max(0, y-radius):y+radius+1, max(0, x-radius):x+radius+1]
            neighborhood_mask = mask[max(0, y-radius):y+radius+1, max(0, x-radius):x+radius+1]
            
            if np.any(~neighborhood_mask):  # 只在存在有效邻域值时才进行插值
                repaired_image[y, x] = 0

        # 将修复后的灰度图像转换回彩色图像
        # height, width = repaired_image.shape

        # # # # 使用双重for循环遍历每个像素
        # for y in range(height):  # 遍历每一行
        #     for x in range(width):  # 遍历每一列
        #         pixel_value = repaired_image[y, x]  # 访问像素值
        #         print(f"Pixel at ({y}, {x}): {pixel_value}")
        repaired_image_color = color.gray2rgb(repaired_image)

        # 显示原图和修复后的图像
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        ax[0].imshow(image)
        ax[0].set_title('Original Image')
        ax[0].axis('off')

        ax[1].imshow(repaired_image_color)
        ax[1].set_title('Repaired Image (Linear Interpolation with 5x5 Neighborhood)')
        ax[1].axis('off')

        plt.show()

        # 保存修复后的图像
        output_path = os.path.join(output_dir, filename)
        io.imsave(output_path, (repaired_image_color * 255).astype(np.uint8))
        print(f"Repaired image saved to {output_path}")