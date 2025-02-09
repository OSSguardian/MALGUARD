import os
import json
import time
# 定义特征抽取的功能
def extract_features(fea_set_path, pack_dir):
    # 加载特征集文件 (fea_set.json)
    with open(fea_set_path, 'r', encoding='utf-8') as f:
        feature_set = json.load(f)

    # 获取特征集中的 API 名单及其顺序
    api_feature_map = {api["api_name"]: 0 for api in feature_set["apis"]}

    # 遍历包的目录
    for package_dir in os.listdir(pack_dir):
        package_path = os.path.join(pack_dir, package_dir)
        try:
            if os.path.isdir(package_path):
                # 每个包的 api_ex.json 文件路径
                api_ex_file = os.path.join(package_path, "katz_new.json")
                if os.path.exists(api_ex_file):
                    # 加载包的 api_ex.json 文件
                    with open(api_ex_file, 'r', encoding='utf-8') as f:
                        api_ex_data = json.load(f)

                    # 初始化一个空的特征向量
                    feature_vector = {api: 0 for api in api_feature_map}

                    # 为每个 API 设置特征值
                    for api_name, feature_value in api_ex_data.items():
                        if api_name in api_feature_map:
                            feature_vector[api_name] = feature_value

                    # 将特征向量保存为 json 文件
                    output_file = os.path.join(package_path, "katz_feature_vector.json")
                    with open(output_file, 'w', encoding='utf-8') as f:
                        json.dump(feature_vector, f, indent=4)

                    # print(f"特征向量已保存到 {output_file}")
        except Exception as e:
            print(f"处理包 {package_path} 时发生错误: {e}")

# 主程序入口
if __name__ == "__main__":

    # 记录当前时间并输出
    starttime = time.time()
    print(f"开始处理时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")
    # 配置文件和目录路径
    fea_set_path = r"E:\py-torch-learning\py-torch-learning\src\social-network\katz_sensitive_api.json"  # fea_set.json 文件路径
    pack_dir = r"F:\weekly_update\1229-0110_dcp"  # 包的根目录路径

    # 调用特征抽取功能
    extract_features(fea_set_path, pack_dir)

    # 输出结束的时间，并计算时间差
    endtime = time.time()
    totaltime = endtime - starttime
    print(f"处理结束，耗时: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")
    print(f"程序结束执行。总用时: {totaltime:.2f} 秒")

