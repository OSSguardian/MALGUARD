import os
import tarfile

def extract_tar_files(input_folder, output_folder):
    # 检查输出文件夹是否存在，如果不存在则创建
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 遍历输入文件夹中的所有文件和子目录
    for root, _, files in os.walk(input_folder):
        for file_name in files:
            # print(f"正在处理文件: {file_name}")
            file_path = os.path.join(root, file_name)
            # 检查文件是否是.tar.gz文件
            if file_name.endswith('.tar.gz'):
                # 解压缩文件
                try:
                    with tarfile.open(file_path, 'r:gz') as tar:
                        # 将文件解压到指定的输出文件夹中
                        tar.extractall(output_folder)
                        print(f"已解压文件: {file_name} 到 {output_folder}")
                except Exception as e:
                    print(f"解压文件 {file_name} 时出错: {e}")

# 示例用法
input_folder = r"F:\weekly_update\1229-0110"  # 替换成包含.tar.gz文件的文件夹路径
output_folder = r"F:\weekly_update\1229-0110_dcp"  # 替换成解压后文件夹的路径
extract_tar_files(input_folder, output_folder)
