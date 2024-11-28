from PIL import Image
import os

def convert_single_channel_to_rgb(input_folder, output_folder):
    # 确保输出文件夹存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 遍历输入文件夹中的所有图像
    for filename in os.listdir(input_folder):
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)

        # 打开图像
        image = Image.open(input_path)

        # 如果是单通道图像，转换为 RGB
        if image.mode == 'L':
            print("convert")
            image = image.convert('RGB')

        # 保存图像
        image.save(output_path)

if __name__ == "__main__":
    input_folder = r"D:\Desktop\data_NoSeg\data\result_level\test\data_0311\test\day5"
    output_folder = r"D:\Desktop\data_NoSeg\data\result_level\test\data_0311\test\day5"

    convert_single_channel_to_rgb(input_folder, output_folder)