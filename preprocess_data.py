"""
加载数据文件包括训练集和测试集的图像数据和对应标签，
对图像数据做标准化处理，
从60000训练数据集中划分出10000作为验证集。
"""

import numpy as np
import gzip

def open_images(filename):
    with gzip.open(filename, 'rb') as file:
        # 跳过前16个字节的头信息
        data = np.frombuffer(file.read(), np.uint8, offset=16)
    return data.reshape(-1, 784)  

def open_labels(filename):
    with gzip.open(filename, 'rb') as file:
        # 跳过前8个字节的头信息
        data = np.frombuffer(file.read(), np.uint8, offset=8)
    return data

def load_local_data():
    train_X = open_images('data/train-images-idx3-ubyte.gz')
    train_y = open_labels('data/train-labels-idx1-ubyte.gz')
    test_X = open_images('data/t10k-images-idx3-ubyte.gz')
    test_y = open_labels('data/t10k-labels-idx1-ubyte.gz')

    # 标准化
    train_X = train_X / 255.0
    test_X = test_X / 255.0

    return train_X, train_y, test_X, test_y

train_X, train_y, test_X, test_y = load_local_data()  

# 划分一部分数据作为验证集
num_validation = 10000
num_train = train_X.shape[0] - num_validation

X_train, y_train = train_X[:num_train], train_y[:num_train]
X_val, y_val = train_X[num_train:], train_y[num_train:]

