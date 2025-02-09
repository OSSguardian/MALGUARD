import matplotlib.pyplot as plt

from data_loader import load_data
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report,confusion_matrix
from tqdm import tqdm
import numpy as np
from lime import lime_tabular


# # 加载数据
# X_train, X_test, y_train, y_test = load_data(
#     r'E:\py-torch-learning\py-torch-learning\src\dataset\mal_katz_feature_vectors.txt',
#     r'E:\py-torch-learning\py-torch-learning\src\dataset\ben_katz_feature_vectors.txt')
X_train, X_test, y_train, y_test = load_data(
    r'E:\py-torch-learning\py-torch-learning\src\social-network\mal_harmonic_feature_vec.txt',
    r'E:\py-torch-learning\py-torch-learning\src\social-network\ben_harmonic_feature_vec.txt')


# 初始化 LIME 解释器
explainer = lime_tabular.LimeTabularExplainer(
    X_train,
    training_labels=y_train,
    feature_names=[f"feature_{i}" for i in range(X_train.shape[1])],
    class_names=["benign", "malicious"],  # 假设标签为良性和恶意
    # verbose=True,
    verbose=False,
    mode="classification"
)


def train_with_progress_bar(model, X_train, y_train, X_test, y_test, model_name, n_iter=10):
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
    tqdm.write(classification_report(y_test, y_pred,digits=5))
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    pre = tp / (tp + fp)
    recall = tp / (tp + fn)
    tqdm.write(f"Precision: {pre}")
    tqdm.write(f"Recall: {recall}")
    # tqdm.write(f"Elapsed Time: {elapsed_time:.5f} seconds")
    tqdm.write(f"Confusion Matrix:\n{cm}")
    tqdm.write(f"TP (True Positives): {tp}")
    tqdm.write(f"FP (False Positives): {fp}")
    tqdm.write(f"TN (True Negatives): {tn}")
    tqdm.write(f"FN (False Negatives): {fn}")

    # 使用 LIME 分析每个测试样本的特征重要性
    # for i, test_sample in enumerate(X_test[:10]):  # 这里仅演示前10个样本
    #     explanation = explainer.explain_instance(test_sample, model.predict_proba, num_features=5)
    #     tqdm.write(f"\nSample {i+1} Explanation:")
    #     # explanation.as_pyplot_figure()
    #     # plt.show()
    #     for feature, weight in explanation.as_list():
    #         tqdm.write(f"{feature}: {weight}")
    #         #print(explanation.as_pyplot_figure())





# Naive Bayes (NB)
nb_model = GaussianNB()
train_with_progress_bar(nb_model, X_train, y_train, X_test, y_test, "Naive Bayes (NB)")

# Multi-Layer Perceptron (MLP)
mlp_model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=1, warm_start=True, random_state=42)
train_with_progress_bar(mlp_model, X_train, y_train, X_test, y_test, "Multi-Layer Perceptron (MLP)", n_iter=300)

# Random Forest (RF)
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
train_with_progress_bar(rf_model, X_train, y_train, X_test, y_test, "Random Forest (RF)", n_iter=10)

# Decision Tree (DT)
dt_model = DecisionTreeClassifier(random_state=42)
train_with_progress_bar(dt_model, X_train, y_train, X_test, y_test, "Decision Tree (DT)")

# SVM替代
# sgd_model = SGDClassifier(loss="hinge", random_state=42)
sgd_model = SGDClassifier(loss="log_loss", random_state=42)
train_with_progress_bar(sgd_model, X_train, y_train, X_test, y_test, "SGD Classifier (SVM)")
