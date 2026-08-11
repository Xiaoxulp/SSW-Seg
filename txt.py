import os

def generate_image_list(image_folder, output_txt_file):
    """
    将文件夹中的所有图片文件名（不含后缀）写入一个文本文件

    Args:
        image_folder (str): 存储图像的文件夹路径
        output_txt_file (str): 输出的文本文件路径
    """
    # 获取所有图片文件的路径
    image_files = [f for f in os.listdir(image_folder) if f.endswith(('.jpg', '.jpeg', '.png'))]

    # 打开文件进行写入
    with open(output_txt_file, 'w') as f:
        for image in image_files:
            # 只获取文件名（不含路径和后缀）
            file_name_without_ext = os.path.splitext(image)[0]
            f.write(file_name_without_ext + "\n")
    print(f"所有图像文件名（不含后缀）已写入 {output_txt_file}")

# 使用示例
image_folder = "/home/liuyameng/data/ISIC/Test/ISBI2016_ISIC_Part1_Test_Data"  # 替换为你的图像文件夹路径
output_txt_file = "/home/liuyameng/data/ISIC/val.txt"  # 替换为输出的文本文件路径

generate_image_list(image_folder, output_txt_file)
