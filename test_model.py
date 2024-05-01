"""
加载保存的模型，在测试集上进行测试
"""

from preprocess_data import *
import numpy as np
from neural_network import *
from train_model import *

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


model = ThreeLayerNN(input_dim=config.get('input_dim'), hidden_dim=config.get('hidden_dim'), output_dim=config.get('output_dim'), activation=config.get('activation'), reg_lambda=config.get('reg_lambda'))

data = np.load('model_weights.npz')

# 加载模型
for key in data.files:
    model.params[key] = data[key]

# 测试集测试
test_accuracy = evaluate(model, test_X, test_y)
print(f"Test Accuracy: {test_accuracy:.2f}")