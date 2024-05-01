"""
经过参数查找后，找到最优参数进行模型训练并保存
"""

from preprocess_data import *
from neural_network import *
from train_model import *
import matplotlib.pyplot as plt
import numpy as np
import json

# 最优模型参数配置
config = {
    'input_dim': 784,
    'hidden_dim': 150,
    'output_dim': 10,
    'learning_rate': 0.01,
    'lr_decay': 0.95,
    'batch_size': 32,
    'num_epochs': 50,
    'reg_lambda': 0.001,
    'activation': 'relu'
}

# 构建模型
best_model = ThreeLayerNN(input_dim=config.get('input_dim'), hidden_dim=config.get('hidden_dim'), output_dim=config.get('output_dim'), activation=config.get('activation'), reg_lambda=config.get('reg_lambda'))

# 模型训练并保存训练集和验证集的loss和accuracy
history = train(best_model, train_X, train_y, X_val, y_val, num_epochs=config.get('num_epochs'), learning_rate=config.get('learning_rate'), batch_size=config.get('batch_size'), lr_decay=config.get('lr_decay'))

# 保存模型权重
np.savez('model_weights.npz', **best_model.params)

# 保存训练集和验证集的loss和accuracy，用于可视化
with open('model_history.json','w') as f:
    json.dump(history, f)

