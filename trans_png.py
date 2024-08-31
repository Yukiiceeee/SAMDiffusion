from PIL import Image

def traverse_non_zero_pixels(image_path):
    # 打开图像
    image = Image.open(image_path)
    # 将图像转换为灰度模式
    image = image.convert('L')
    # 获取图像的像素数据
    pixels = image.load()
    # print(111)
    # 遍历图像的每个像素
    width, height = image.size
    for y in range(height):
        for x in range(width):
            pixel_value = pixels[x, y]
            if pixel_value != 0:
                print(f'Non-zero pixel at ({x}, {y}): {pixel_value}')

# 使用示例
image_path = '/home/zhuyifan/Cyan_A40/sam_data/VOC_Multi_Attention_aeroplane_sub_4000_NoClipRetrieval_sample/ground_truth/image_aeroplane_200113_mask.png'
traverse_non_zero_pixels(image_path)