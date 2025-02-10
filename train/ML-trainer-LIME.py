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


def get_malicious_purposes(api_id, sensitive_apis):
    for api in sensitive_apis:
        if api["api_id"] == api_id:
            return api["malicious_purposes"]
    return ["unknown"]


def get_malicious_api_name(api_id, sensitive_apis):
    for api in sensitive_apis:
        if api["api_id"] == api_id:
            return api["api_name"]
    return ["unknown"]


def load_sensitive_apis(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)["apis"]


def train_with_progress_bar(model, X_train, y_train, X_test, y_test, model_name, model_save_path, sensitive_apis,
                            n_iter=100):
    model_dir = os.path.join(model_save_path, model_name.replace(" ", "_").lower())
    os.makedirs(model_dir, exist_ok=True)

    with tqdm(total=n_iter, desc=f"Training {model_name}", unit="iter") as pbar:
        for i in range(n_iter):
            if hasattr(model, 'partial_fit'):
                model.partial_fit(X_train, y_train, classes=np.unique(y_train))
            else:
                model.fit(X_train, y_train)
            pbar.update(1)

    y_pred = model.predict(X_test)
    tqdm.write(f"\n{model_name} Results")

    report = classification_report(y_test, y_pred, digits=5)
    tqdm.write(report)
    #with open(os.path.join(model_dir, f"{model_name.replace(' ', '_').lower()}_report.txt"), "w") as f:
        #f.write(report)


    # model_filename = os.path.join(model_dir, f"{model_name.replace(' ', '_').lower()}_model.pkl")
    # joblib.dump(model, model_filename)
    # tqdm.write(f"{model_name} saved to {model_filename}")

    # malicious_indices = np.where(y_pred == 1)[0]  
    # for i in malicious_indices: 
    #     test_sample = X_test[i]
    #     explanation = explainer.explain_instance(test_sample, model.predict_proba, num_features=5)
    #     tqdm.write(f"\nSample {i + 1} (Predicted as Malicious) Explanation:")
    #     for feature, weight in explanation.as_list():
    #         if weight != 0:  
    #             print(feature)
    #             match = re.search(r"feature_(\d+)", feature)
    #             print(match)
    #             feature_idx = int(match.group(1))
    #             print(feature_idx)
    #             api_id = feature_idx + 1 
    #             malicious_purposes = get_malicious_purposes(api_id, sensitive_apis)
    #             API_NAME = get_malicious_api_name(api_id,sensitive_apis)
    #             tqdm.write(f"{feature}: {weight}, API NAME :{API_NAME}, API ID: {api_id}, Malicious Purposes: {malicious_purposes}")

def main():
    mal_data_path = r'mal_closeness_feature_vectors.txt'
    ben_data_path = r'ben_closeness_feature_vectors.txt'
    sensitive_api_file = r"closeness_sensitive_api.json"
    model_save_path = r"model\closeness"

    X_train, X_test, y_train, y_test = load_data(
        mal_data_path,
        ben_data_path)

    # with open(sensitive_api_file, "r") as f:
    #     sensitive_apis = json.load(f)
    sensitive_apis = load_sensitive_apis(sensitive_api_file) 

    global explainer
    explainer = lime_tabular.LimeTabularExplainer(
        X_train,
        training_labels=y_train,
        feature_names=[f"feature_{i}" for i in range(X_train.shape[1])],
        class_names=["benign", "malicious"], 
        verbose=False,
        mode="classification"
    )

    nb_model = GaussianNB()
    train_with_progress_bar(nb_model, X_train, y_train, X_test, y_test, "Naive Bayes (NB)", model_save_path,
                            sensitive_apis, n_iter=100)


    mlp_model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=1, warm_start=True, random_state=42)
    train_with_progress_bar(mlp_model, X_train, y_train, X_test, y_test, "Multi-Layer Perceptron (MLP)",
                            model_save_path, sensitive_apis, n_iter=500)

    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    train_with_progress_bar(rf_model, X_train, y_train, X_test, y_test, "Random Forest (RF)", model_save_path,
                            sensitive_apis, n_iter=100)

    dt_model = DecisionTreeClassifier(random_state=42)
    train_with_progress_bar(dt_model, X_train, y_train, X_test, y_test, "Decision Tree (DT)", model_save_path,
                            sensitive_apis, n_iter=100)


    sgd_model = SGDClassifier(loss="log_loss", random_state=42)
    train_with_progress_bar(sgd_model, X_train, y_train, X_test, y_test, "SGD Classifier (SVM)", model_save_path,
                            sensitive_apis, n_iter=100)


def main1():

    mal_data_path = r'mal_degree_feature_vectors.txt'
    ben_data_path = r'ben_degree_feature_vectors.txt'
    sensitive_api_file = r"degree_sensitive_api.json"
    model_save_path = r"model\degree"

    X_train, X_test, y_train, y_test = load_data(
        mal_data_path,
        ben_data_path)

    # with open(sensitive_api_file, "r") as f:
    #     sensitive_apis = json.load(f)
    sensitive_apis = load_sensitive_apis(sensitive_api_file) 

    global explainer
    explainer = lime_tabular.LimeTabularExplainer(
        X_train,
        training_labels=y_train,
        feature_names=[f"feature_{i}" for i in range(X_train.shape[1])],
        class_names=["benign", "malicious"], 
        verbose=False,
        mode="classification"
    )


    nb_model = GaussianNB()
    train_with_progress_bar(nb_model, X_train, y_train, X_test, y_test, "Naive Bayes (NB)", model_save_path,
                            sensitive_apis, n_iter=100)

    mlp_model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=1, warm_start=True, random_state=42)
    train_with_progress_bar(mlp_model, X_train, y_train, X_test, y_test, "Multi-Layer Perceptron (MLP)",
                            model_save_path, sensitive_apis, n_iter=500)

    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    train_with_progress_bar(rf_model, X_train, y_train, X_test, y_test, "Random Forest (RF)", model_save_path,
                            sensitive_apis, n_iter=100)

    dt_model = DecisionTreeClassifier(random_state=42)
    train_with_progress_bar(dt_model, X_train, y_train, X_test, y_test, "Decision Tree (DT)", model_save_path,
                            sensitive_apis, n_iter=100)

    sgd_model = SGDClassifier(loss="log_loss", random_state=42)
    train_with_progress_bar(sgd_model, X_train, y_train, X_test, y_test, "SGD Classifier (SVM)", model_save_path,
                            sensitive_apis, n_iter=100)


def main2():
    mal_data_path = r'mal_harmonic_feature_vectors.txt'
    ben_data_path = r'ben_harmonic_feature_vectors.txt'
    sensitive_api_file = r"harmonic_sensitive_api.json"
    model_save_path = r"model\harmonic"

    X_train, X_test, y_train, y_test = load_data(
        mal_data_path,
        ben_data_path)

    # with open(sensitive_api_file, "r") as f:
    #     sensitive_apis = json.load(f)
    sensitive_apis = load_sensitive_apis(sensitive_api_file)  


    global explainer
    explainer = lime_tabular.LimeTabularExplainer(
        X_train,
        training_labels=y_train,
        feature_names=[f"feature_{i}" for i in range(X_train.shape[1])],
        class_names=["benign", "malicious"],
        verbose=False,
        mode="classification"
    )

    nb_model = GaussianNB()
    train_with_progress_bar(nb_model, X_train, y_train, X_test, y_test, "Naive Bayes (NB)", model_save_path,
                            sensitive_apis, n_iter=100)

    mlp_model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=1, warm_start=True, random_state=42)
    train_with_progress_bar(mlp_model, X_train, y_train, X_test, y_test, "Multi-Layer Perceptron (MLP)",
                            model_save_path, sensitive_apis, n_iter=500)

    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    train_with_progress_bar(rf_model, X_train, y_train, X_test, y_test, "Random Forest (RF)", model_save_path,
                            sensitive_apis, n_iter=100)

    dt_model = DecisionTreeClassifier(random_state=42)
    train_with_progress_bar(dt_model, X_train, y_train, X_test, y_test, "Decision Tree (DT)", model_save_path,
                            sensitive_apis, n_iter=100)

    sgd_model = SGDClassifier(loss="log_loss", random_state=42)
    train_with_progress_bar(sgd_model, X_train, y_train, X_test, y_test, "SGD Classifier (SVM)", model_save_path,
                            sensitive_apis, n_iter=100)

def main3():
    mal_data_path = r'mal_katz_feature_vectors.txt'
    ben_data_path = r'ben_katz_feature_vectors.txt'
    sensitive_api_file = r"katz_sensitive_api.json"
    model_save_path = r"model\katz"

    X_train, X_test, y_train, y_test = load_data(
        mal_data_path,
        ben_data_path)

    # with open(sensitive_api_file, "r") as f:
    #     sensitive_apis = json.load(f)
    sensitive_apis = load_sensitive_apis(sensitive_api_file)

    global explainer
    explainer = lime_tabular.LimeTabularExplainer(
        X_train,
        training_labels=y_train,
        feature_names=[f"feature_{i}" for i in range(X_train.shape[1])],
        class_names=["benign", "malicious"],  
        verbose=False,
        mode="classification"
    )

    nb_model = GaussianNB()
    train_with_progress_bar(nb_model, X_train, y_train, X_test, y_test, "Naive Bayes (NB)", model_save_path,
                            sensitive_apis, n_iter=100)

    mlp_model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=1, warm_start=True, random_state=42)
    train_with_progress_bar(mlp_model, X_train, y_train, X_test, y_test, "Multi-Layer Perceptron (MLP)",
                            model_save_path, sensitive_apis, n_iter=500)

    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    train_with_progress_bar(rf_model, X_train, y_train, X_test, y_test, "Random Forest (RF)", model_save_path,
                            sensitive_apis, n_iter=100)

    dt_model = DecisionTreeClassifier(random_state=42)
    train_with_progress_bar(dt_model, X_train, y_train, X_test, y_test, "Decision Tree (DT)", model_save_path,
                            sensitive_apis, n_iter=100)

    sgd_model = SGDClassifier(loss="log_loss", random_state=42)
    train_with_progress_bar(sgd_model, X_train, y_train, X_test, y_test, "SGD Classifier (SVM)", model_save_path,
                            sensitive_apis, n_iter=100)



if __name__ == "__main__":
    main()
    main1()
    main2()
    main3()
