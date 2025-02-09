import numpy as np
from sklearn.model_selection import train_test_split


def load_data(mal_file, ben_file):
    """
    加载恶意包和良性包的特征向量数据，并划分为训练集和测试集。

    Parameters:
        mal_file (str): 恶意包特征向量文件路径
        ben_file (str): 良性包特征向量文件路径

    Returns:
        X_train, X_test, y_train, y_test: 训练集和测试集的特征和标签
    """
    mal_data = np.loadtxt(mal_file)
    ben_data = np.loadtxt(ben_file)

    # 合并良性和恶性数据
    data = np.vstack((mal_data, ben_data))

    # 划分特征和标签
    X = data[:, :-1]  # 特征向量
    y = data[:, -1]  # 标签 (1: 恶意包, 0: 良性包)

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test
