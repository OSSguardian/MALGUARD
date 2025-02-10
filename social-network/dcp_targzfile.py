import os
import tarfile

def extract_tar_files(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for root, _, files in os.walk(input_folder):
        for file_name in files:
        
            file_path = os.path.join(root, file_name)
            if file_name.endswith('.tar.gz'):
                try:
                    with tarfile.open(file_path, 'r:gz') as tar:
                        tar.extractall(output_folder)
                        print(f"已解压文件: {file_name} 到 {output_folder}")
                except Exception as e:
                    print(f"解压文件 {file_name} 时出错: {e}")

input_folder = r"weekly_update\1229-0110"  
output_folder = r"weekly_update\1229-0110_dcp" 
extract_tar_files(input_folder, output_folder)
