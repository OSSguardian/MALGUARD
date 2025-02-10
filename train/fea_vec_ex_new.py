import os
import json
import time

def extract_features(fea_set_path, pack_dir):
    with open(fea_set_path, 'r', encoding='utf-8') as f:
        feature_set = json.load(f)
    api_feature_map = {api["api_name"]: 0 for api in feature_set["apis"]}
    for package_dir in os.listdir(pack_dir):
        package_path = os.path.join(pack_dir, package_dir)
        try:
            if os.path.isdir(package_path):
                api_ex_file = os.path.join(package_path, "katz_new.json")
                if os.path.exists(api_ex_file):
                    with open(api_ex_file, 'r', encoding='utf-8') as f:
                        api_ex_data = json.load(f)
                    feature_vector = {api: 0 for api in api_feature_map}

                    for api_name, feature_value in api_ex_data.items():
                        if api_name in api_feature_map:
                            feature_vector[api_name] = feature_value

                    output_file = os.path.join(package_path, "katz_feature_vector.json")
                    with open(output_file, 'w', encoding='utf-8') as f:
                        json.dump(feature_vector, f, indent=4)

        except Exception as e:
            print(f"处理包 {package_path} 时发生错误: {e}")

if __name__ == "__main__":
    starttime = time.time()
    print(f"开始处理时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")
    fea_set_path = r"katz_sensitive_api.json" 
    pack_dir = r"weekly_update\1229-0110_dcp"  
    extract_features(fea_set_path, pack_dir)
    endtime = time.time()
    totaltime = endtime - starttime
    print(f"处理结束，耗时: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")
    print(f"程序结束执行。总用时: {totaltime:.2f} 秒")

