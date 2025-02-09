import os
import json
import time

import joblib
import numpy as np
import re
from tqdm import tqdm
from lime import lime_tabular
from data_loader import load_data

mal_data_path = r'E:\py-torch-learning\py-torch-learning\src\dataset\mal_katz_feature_vectors.txt'
ben_data_path = r'E:\py-torch-learning\py-torch-learning\src\dataset\ben_katz_feature_vectors.txt'

X_train, X_test, y_train, y_test = load_data(
        mal_data_path,
        ben_data_path)
def get_malicious_purposes(api_name, sensitive_apis):
    # 遍历敏感API列表，根据 api_id 查找
    for api in sensitive_apis:
        if api["api_name"] == api_name:
            return api["malicious_purposes"]
    return ["unknown"]




def load_sensitive_apis(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)["apis"]


def parse_cfg_file(cfg_path):
    """
    解析 CFG.txt 文件，提取 API 名及其对应的文件名、行号和函数。
    :param cfg_path: CFG.txt 文件路径
    :return: List of tuples, 每个元素为 (api_name, file_name, line_number, function_name)
    """
    if not os.path.exists(cfg_path):
        return []  # 如果文件不存在，返回空列表

    results = []
    with open(cfg_path, 'r', encoding='utf-8') as f:
        for line in f:
            match = re.match(r"<(.+?) (\d+) (.+?)>\s+(.+)", line.strip())
            if match:
                file_name, line_number, function_name, api_name = match.groups()
                results.append((file_name, int(line_number), api_name, function_name))
    return results

def evaluate_packages_with_lime(test_dir, model_save_path, sensitive_api_file):
    # 加载敏感 API 数据
    sensitive_apis = load_sensitive_apis(sensitive_api_file)

    # 加载模型
    models = {
        # "Naive Bayes (NB)": os.path.join(model_save_path, "naive_bayes_(nb)", "naive_bayes_(nb)_model.pkl"),
        "Random Forest (RF)": os.path.join(model_save_path, "random_forest_(rf)", "random_forest_(rf)_model.pkl"),
        "Decision Tree (DT)": os.path.join(model_save_path, "decision_tree_(dt)", "decision_tree_(dt)_model.pkl"),
        "SGD Classifier (SVM)": os.path.join(model_save_path, "sgd_classifier_(svm)", "sgd_classifier_(svm)_model.pkl"),
        "Multi-Layer Perceptron (MLP)": os.path.join(model_save_path, "multi-layer_perceptron_(mlp)",
                                                     "multi-layer_perceptron_(mlp)_model.pkl")
    }

    models = {name: joblib.load(path) for name, path in models.items()}

    # 初始化 LIME 解释器
    explainer = None  # 延迟初始化以便动态加载训练数据

    # 遍历测试目录中的包
    malicious_packages = {model_name: [] for model_name in models}  # 每个模型单独保存恶意包名称

    for package_name in tqdm(os.listdir(test_dir), desc="Evaluating packages", unit="package"):
        package_path = os.path.join(test_dir, package_name)
        fea_vec_file = os.path.join(package_path, "katz_feature_vector.json")

        explanation_files = {
            model_name: os.path.join(package_path, f"explain_{model_name.replace(' ', '_').lower()}_katz.txt") for model_name
            in models}

        if not os.path.isfile(fea_vec_file):
            tqdm.write(f"Skipping {package_name}: katz_feature_vector.json not found.")
            continue

        with open(fea_vec_file, "r") as f:
            feature_vector = json.load(f)

        # 转换为模型输入格式
        feature_name = list(feature_vector.keys())
        features = np.array([list(feature_vector.values())])
        # print(feature_name)
        # print(features)

        for model_name, model in models.items():
            # 第一次处理时初始化 LIME 解释器
            if explainer is None:
                explainer = lime_tabular.LimeTabularExplainer(
                    X_train,
                    training_labels=y_train,
                    feature_names=list(feature_vector.keys()),
                    class_names=["benign", "malicious"],
                    verbose=False,
                    mode="classification"
                )

            # 预测结果
            prediction = model.predict(features)[0]
            total_fea = np.sum(features)
            if prediction == 1 and total_fea > 0:  # 被预测为恶意
                malicious_packages[model_name].append(package_name)

                # 生成 LIME 解释
                explanation = explainer.explain_instance(features[0], model.predict_proba, num_features=len(feature_vector))
                cfg_path = os.path.join(package_path, "CFG.txt")
                cfg_entries = parse_cfg_file(cfg_path)

                with open(explanation_files[model_name], "w", encoding='utf-8') as exp_file:
                    exp_file.write(f"LIME Explanation for package {package_name}:\n\n\n")
                    print(f"LIME Explanation for package {package_name}:")
                    cout_num = 0
                    result = []

                    for feature, weight in explanation.as_list():
                        match = re.match(r'^[^><]*', feature)
                        match = match.group().strip()
                        # feature_value = feature_vector.get(feature, 0)
                        for key, value in feature_vector.items():
                            if key == match:
                                feature_value = value
                        if feature_value != 0 and weight != 0 and cout_num <= 5:
                            for file_name, line_number, api_name, function_name in cfg_entries:
                                if match == api_name:
                                    print(match)
                                    print(file_name, line_number, api_name, function_name)
                                    malicious_purposes = get_malicious_purposes(match, sensitive_apis)
                                    cout_num += 1
                                    result.append((file_name, line_number, api_name, function_name, malicious_purposes))
                                    # print(result)
                            # exp_file.write(f"{match}: {feature_value},\nAPI NAME:  {match},\nMalicious Purposes:\n{malicious_purposes}\n\n\n")
                            # print(f"{feature}: {weight}, API NAME: {feature}, Malicious Purposes: {malicious_purposes}")
                    result = sorted(result)
                    for file_name, line_number, api_name, function_name, malicious_purposes in result:
                        exp_file.write(f"In file {file_name} line {line_number}, the package holder use the sensitive api:\n[{api_name}],\nin function/global {function_name},\nwhich may be used for:\n{malicious_purposes}\n\n\n")
                        # print(f"In file {file_name} line {line_number}, the package holder use the sensitive api:\n[{api_name}],\nin function/global [{function_name}],\nwhich may be used for:\n{malicious_purposes}\n\n\n")
                        # print(f"{file_name}:{line_number}  {api_name}  {function_name}  {malicious_purposes}")

    # 保存恶意包列表
    for model_name, packages in malicious_packages.items():
        malicious_file = os.path.join(test_dir, f"malicious_{model_name.replace(' ', '_').lower()}_katz.txt")
        with open(malicious_file, "w") as f:
            f.write("\n".join(packages))
        tqdm.write(f"Malicious packages for {model_name} saved to {malicious_file}")

def main():
    # 你的模型保存路径和测试目录路径
    test_dir = r"F:\weekly_update\1229-0110_dcp"
    model_save_path = r"E:\py-torch-learning\py-torch-learning\src\model\katz"
    sensitive_api_file = r"E:\py-torch-learning\py-torch-learning\src\social-network\katz_sensitive_api.json"

    evaluate_packages_with_lime(test_dir, model_save_path, sensitive_api_file)


if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    print(f"程序结束执行。总用时: {end_time - start_time:.2f} 秒")

