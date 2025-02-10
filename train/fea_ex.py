import os
import json
from tqdm import tqdm
import numpy as np


def load_top_features(centrality_file, top_n=500):
    with open(centrality_file, 'r', encoding='utf-8') as f:
        centrality_data = json.load(f)

    top_features = list(centrality_data.keys())[:top_n]
    top_centrality_values = {k: centrality_data[k] for k in top_features}

    return top_centrality_values


def extract_feature_vector(package_dir, top_features):
    feature_vector = np.zeros(len(top_features))

    for root, _, files in os.walk(package_dir):
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        for i, feature in enumerate(top_features):
                            if feature in content:
                                feature_vector[i] += 1
                except Exception as e:
                    tqdm.write(f"Error processing file: {file_path}. Error: {e}")
                    continue

    for i, feature in enumerate(top_features):
        feature_vector[i] *= top_features[feature]

    return feature_vector


def save_feature_vector(feature_vector, output_file):
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(feature_vector.tolist(), f, indent=4, ensure_ascii=False)


def main(base_dir, centrality_file, top_n=500):
    top_features = load_top_features(centrality_file, top_n)
    package_dirs = [os.path.join(base_dir, d) for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]

    error_log = []

    for package_dir in tqdm(package_dirs, desc="Extracting features from packages"):
        try:
            feature_vector = extract_feature_vector(package_dir, top_features)

            # output_file = os.path.join(package_dir, 'mal_harmonic_feature_vector.json')
            # output_file = os.path.join(package_dir, 'mal_katz_feature_vector.json')
            # output_file = os.path.join(package_dir, 'ben_degree_feature_vector.json')
            output_file = os.path.join(package_dir, 'mal_closeness_feature_vector.json')
            save_feature_vector(feature_vector, output_file)
        except Exception as e:
            error_msg = f"Error processing package: {package_dir}. Error: {e}"
            tqdm.write(error_msg)
            error_log.append(error_msg)
    if error_log:
        with open('ben_error_log.txt', 'w', encoding='utf-8') as f:
            for error in error_log:
                f.write(f"{error}\n")

# top n = 200, 300, 400, 500
main(r'', 'closeness_centrality.json', top_n=200)
