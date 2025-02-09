import os
import json
import matplotlib.pyplot as plt
from data_loader import load_data
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import numpy as np
from lime import lime_tabular
import joblib
import re


# 输出API的malicious_purposes
def get_malicious_purposes(api_id, sensitive_apis):
    # 遍历敏感API列表，根据 api_id 查找
    for api in sensitive_apis:
        if api["api_id"] == api_id:
            return api["malicious_purposes"]
    return ["unknown"]


def get_malicious_api_name(api_id, sensitive_apis):
    # 遍历敏感API列表，根据 api_id 查找
    for api in sensitive_apis:
        if api["api_id"] == api_id:
            return api["api_name"]
    return ["unknown"]


def load_sensitive_apis(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)["apis"]


# 训练并保存模型的函数
def train_with_progress_bar(model, X_train, y_train, X_test, y_test, model_name, model_save_path, sensitive_apis,
                            n_iter=100):
    # 确保保存模型的目录存在
    model_dir = os.path.join(model_save_path, model_name.replace(" ", "_").lower())
    os.makedirs(model_dir, exist_ok=True)

    # 进度条显示
    with tqdm(total=n_iter, desc=f"Training {model_name}", unit="iter") as pbar:
        for i in range(n_iter):
            if hasattr(model, 'partial_fit'):
                model.partial_fit(X_train, y_train, classes=np.unique(y_train))
            else:
                # 对于不支持 partial_fit 的模型，使用全部数据进行 fit
                model.fit(X_train, y_train)
            pbar.update(1)

    # 预测
    y_pred = model.predict(X_test)
    tqdm.write(f"\n{model_name} Results")

    # 输出精确度、召回率、F1得分等指标，保留五位小数
    report = classification_report(y_test, y_pred, digits=5)
    tqdm.write(report)
    # 将report写入指定txt文件中去
    #with open(os.path.join(model_dir, f"{model_name.replace(' ', '_').lower()}_report.txt"), "w") as f:
        #f.write(report)


    # 保存训练好的模型
    # model_filename = os.path.join(model_dir, f"{model_name.replace(' ', '_').lower()}_model.pkl")
    # joblib.dump(model, model_filename)
    # tqdm.write(f"{model_name} saved to {model_filename}")

    # 只对被预测为 1 (恶意) 的样本进行 LIME 分析
    # malicious_indices = np.where(y_pred == 1)[0]  # 获取被预测为恶意样本的索引
    # for i in malicious_indices:  # 遍历所有恶意样本
    #     test_sample = X_test[i]
    #     explanation = explainer.explain_instance(test_sample, model.predict_proba, num_features=5)
    #     tqdm.write(f"\nSample {i + 1} (Predicted as Malicious) Explanation:")
    #     for feature, weight in explanation.as_list():
    #         if weight != 0:  # 只输出非零特征
    #             print(feature)
    #             match = re.search(r"feature_(\d+)", feature)
    #             print(match)
    #             feature_idx = int(match.group(1))
    #             print(feature_idx)# 获取特征位置
    #             api_id = feature_idx + 1  # 从1开始计数
    #             malicious_purposes = get_malicious_purposes(api_id, sensitive_apis)
    #             API_NAME = get_malicious_api_name(api_id,sensitive_apis)
    #             tqdm.write(f"{feature}: {weight}, API NAME :{API_NAME}, API ID: {api_id}, Malicious Purposes: {malicious_purposes}")

def main():
    # 输入路径
    mal_data_path = r'E:\py-torch-learning\py-torch-learning\src\dataset\mal_closeness_feature_vectors.txt'
    ben_data_path = r'E:\py-torch-learning\py-torch-learning\src\dataset\ben_closeness_feature_vectors.txt'
    sensitive_api_file = r"E:\py-torch-learning\py-torch-learning\src\social-network\closeness_sensitive_api.json"
    model_save_path = r"E:\py-torch-learning\py-torch-learning\src\model\closeness"

    X_train, X_test, y_train, y_test = load_data(
        mal_data_path,
        ben_data_path)

    # 加载 sensitive_api.json
    # with open(sensitive_api_file, "r") as f:
    #     sensitive_apis = json.load(f)
    sensitive_apis = load_sensitive_apis(sensitive_api_file)  # 请根据实际路径修改文件路径

    # 初始化 LIME 解释器
    global explainer
    explainer = lime_tabular.LimeTabularExplainer(
        X_train,
        training_labels=y_train,
        feature_names=[f"feature_{i}" for i in range(X_train.shape[1])],
        class_names=["benign", "malicious"],  # 假设标签为良性和恶意
        verbose=False,
        mode="classification"
    )

    # Naive Bayes (NB)
    nb_model = GaussianNB()
    train_with_progress_bar(nb_model, X_train, y_train, X_test, y_test, "Naive Bayes (NB)", model_save_path,
                            sensitive_apis, n_iter=100)

    # Multi-Layer Perceptron (MLP)
    mlp_model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=1, warm_start=True, random_state=42)
    train_with_progress_bar(mlp_model, X_train, y_train, X_test, y_test, "Multi-Layer Perceptron (MLP)",
                            model_save_path, sensitive_apis, n_iter=500)

    # Random Forest (RF)
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    train_with_progress_bar(rf_model, X_train, y_train, X_test, y_test, "Random Forest (RF)", model_save_path,
                            sensitive_apis, n_iter=100)

    # Decision Tree (DT)
    dt_model = DecisionTreeClassifier(random_state=42)
    train_with_progress_bar(dt_model, X_train, y_train, X_test, y_test, "Decision Tree (DT)", model_save_path,
                            sensitive_apis, n_iter=100)

    # SGD Classifier (SVM替代)
    sgd_model = SGDClassifier(loss="log_loss", random_state=42)
    train_with_progress_bar(sgd_model, X_train, y_train, X_test, y_test, "SGD Classifier (SVM)", model_save_path,
                            sensitive_apis, n_iter=100)


# 主函数
def main1():
    # 输入路径
    mal_data_path = r'E:\py-torch-learning\py-torch-learning\src\dataset\mal_degree_feature_vectors.txt'
    ben_data_path = r'E:\py-torch-learning\py-torch-learning\src\dataset\ben_degree_feature_vectors.txt'
    sensitive_api_file = r"E:\py-torch-learning\py-torch-learning\src\social-network\degree_sensitive_api.json"
    model_save_path = r"E:\py-torch-learning\py-torch-learning\src\model\degree"

    X_train, X_test, y_train, y_test = load_data(
        mal_data_path,
        ben_data_path)

    # 加载 sensitive_api.json
    # with open(sensitive_api_file, "r") as f:
    #     sensitive_apis = json.load(f)
    sensitive_apis = load_sensitive_apis(sensitive_api_file)  # 请根据实际路径修改文件路径

    # 初始化 LIME 解释器
    global explainer
    explainer = lime_tabular.LimeTabularExplainer(
        X_train,
        training_labels=y_train,
        feature_names=[f"feature_{i}" for i in range(X_train.shape[1])],
        class_names=["benign", "malicious"],  # 假设标签为良性和恶意
        verbose=False,
        mode="classification"
    )

    # Naive Bayes (NB)
    nb_model = GaussianNB()
    train_with_progress_bar(nb_model, X_train, y_train, X_test, y_test, "Naive Bayes (NB)", model_save_path,
                            sensitive_apis, n_iter=100)

    # Multi-Layer Perceptron (MLP)
    mlp_model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=1, warm_start=True, random_state=42)
    train_with_progress_bar(mlp_model, X_train, y_train, X_test, y_test, "Multi-Layer Perceptron (MLP)",
                            model_save_path, sensitive_apis, n_iter=500)

    # Random Forest (RF)
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    train_with_progress_bar(rf_model, X_train, y_train, X_test, y_test, "Random Forest (RF)", model_save_path,
                            sensitive_apis, n_iter=100)

    # Decision Tree (DT)
    dt_model = DecisionTreeClassifier(random_state=42)
    train_with_progress_bar(dt_model, X_train, y_train, X_test, y_test, "Decision Tree (DT)", model_save_path,
                            sensitive_apis, n_iter=100)

    # SGD Classifier (SVM替代)
    sgd_model = SGDClassifier(loss="log_loss", random_state=42)
    train_with_progress_bar(sgd_model, X_train, y_train, X_test, y_test, "SGD Classifier (SVM)", model_save_path,
                            sensitive_apis, n_iter=100)


def main2():
    # 输入路径
    mal_data_path = r'E:\py-torch-learning\py-torch-learning\src\dataset\mal_harmonic_feature_vectors.txt'
    ben_data_path = r'E:\py-torch-learning\py-torch-learning\src\dataset\ben_harmonic_feature_vectors.txt'
    sensitive_api_file = r"E:\py-torch-learning\py-torch-learning\src\social-network\harmonic_sensitive_api.json"
    model_save_path = r"E:\py-torch-learning\py-torch-learning\src\model\harmonic"

    X_train, X_test, y_train, y_test = load_data(
        mal_data_path,
        ben_data_path)

    # 加载 sensitive_api.json
    # with open(sensitive_api_file, "r") as f:
    #     sensitive_apis = json.load(f)
    sensitive_apis = load_sensitive_apis(sensitive_api_file)  # 请根据实际路径修改文件路径

    # 初始化 LIME 解释器
    global explainer
    explainer = lime_tabular.LimeTabularExplainer(
        X_train,
        training_labels=y_train,
        feature_names=[f"feature_{i}" for i in range(X_train.shape[1])],
        class_names=["benign", "malicious"],  # 假设标签为良性和恶意
        verbose=False,
        mode="classification"
    )

    # Naive Bayes (NB)
    nb_model = GaussianNB()
    train_with_progress_bar(nb_model, X_train, y_train, X_test, y_test, "Naive Bayes (NB)", model_save_path,
                            sensitive_apis, n_iter=100)

    # Multi-Layer Perceptron (MLP)
    mlp_model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=1, warm_start=True, random_state=42)
    train_with_progress_bar(mlp_model, X_train, y_train, X_test, y_test, "Multi-Layer Perceptron (MLP)",
                            model_save_path, sensitive_apis, n_iter=500)

    # Random Forest (RF)
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    train_with_progress_bar(rf_model, X_train, y_train, X_test, y_test, "Random Forest (RF)", model_save_path,
                            sensitive_apis, n_iter=100)

    # Decision Tree (DT)
    dt_model = DecisionTreeClassifier(random_state=42)
    train_with_progress_bar(dt_model, X_train, y_train, X_test, y_test, "Decision Tree (DT)", model_save_path,
                            sensitive_apis, n_iter=100)

    # SGD Classifier (SVM替代)
    sgd_model = SGDClassifier(loss="log_loss", random_state=42)
    train_with_progress_bar(sgd_model, X_train, y_train, X_test, y_test, "SGD Classifier (SVM)", model_save_path,
                            sensitive_apis, n_iter=100)

def main3():
    # 输入路径
    mal_data_path = r'E:\py-torch-learning\py-torch-learning\src\dataset\mal_katz_feature_vectors.txt'
    ben_data_path = r'E:\py-torch-learning\py-torch-learning\src\dataset\ben_katz_feature_vectors.txt'
    sensitive_api_file = r"E:\py-torch-learning\py-torch-learning\src\social-network\katz_sensitive_api.json"
    model_save_path = r"E:\py-torch-learning\py-torch-learning\src\model\katz"

    X_train, X_test, y_train, y_test = load_data(
        mal_data_path,
        ben_data_path)

    # 加载 sensitive_api.json
    # with open(sensitive_api_file, "r") as f:
    #     sensitive_apis = json.load(f)
    sensitive_apis = load_sensitive_apis(sensitive_api_file)  # 请根据实际路径修改文件路径

    # 初始化 LIME 解释器
    global explainer
    explainer = lime_tabular.LimeTabularExplainer(
        X_train,
        training_labels=y_train,
        feature_names=[f"feature_{i}" for i in range(X_train.shape[1])],
        class_names=["benign", "malicious"],  # 假设标签为良性和恶意
        verbose=False,
        mode="classification"
    )

    # Naive Bayes (NB)
    nb_model = GaussianNB()
    train_with_progress_bar(nb_model, X_train, y_train, X_test, y_test, "Naive Bayes (NB)", model_save_path,
                            sensitive_apis, n_iter=100)

    # Multi-Layer Perceptron (MLP)
    mlp_model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=1, warm_start=True, random_state=42)
    train_with_progress_bar(mlp_model, X_train, y_train, X_test, y_test, "Multi-Layer Perceptron (MLP)",
                            model_save_path, sensitive_apis, n_iter=500)

    # Random Forest (RF)
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    train_with_progress_bar(rf_model, X_train, y_train, X_test, y_test, "Random Forest (RF)", model_save_path,
                            sensitive_apis, n_iter=100)

    # Decision Tree (DT)
    dt_model = DecisionTreeClassifier(random_state=42)
    train_with_progress_bar(dt_model, X_train, y_train, X_test, y_test, "Decision Tree (DT)", model_save_path,
                            sensitive_apis, n_iter=100)

    # SGD Classifier (SVM替代)
    sgd_model = SGDClassifier(loss="log_loss", random_state=42)
    train_with_progress_bar(sgd_model, X_train, y_train, X_test, y_test, "SGD Classifier (SVM)", model_save_path,
                            sensitive_apis, n_iter=100)



if __name__ == "__main__":
    main()
    main1()
    main2()
    main3()
