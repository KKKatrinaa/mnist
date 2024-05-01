"""
超参数查找，调节学习率、隐藏层大小、正则化强度等超参数，找到最优参数
"""

from preprocess_data import *
from neural_network import *
from train_model import *
import numpy as np

def hyperparameter_tuning(X_train, y_train, X_val, y_val):
    # 网格搜索，在不同超参数下进行模型训练，找到最优参数
    learning_rates = [0.01, 0.001]
    hidden_dims = [100, 150]
    reg_lambdas = [0.01, 0.001]

    best_hyperparams = None
    best_val_acc = 0

    for lr in learning_rates:
        for hidden_dim in hidden_dims:
            for reg_lambda in reg_lambdas:
                print(f"Training with lr={lr}, hidden_dim={hidden_dim}, reg_lambda={reg_lambda}")
                model = ThreeLayerNN(input_dim=784, hidden_dim=hidden_dim, output_dim=10, reg_lambda=reg_lambda)
                train(model, X_train, y_train, X_val, y_val, num_epochs=50, learning_rate=lr)
                val_acc = evaluate(model, X_val, y_val)
                print(f"Validation Accuracy: {val_acc:.2f}")

                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_hyperparams = (lr, hidden_dim, reg_lambda)

    print(f"Best hyperparameters: Learning Rate={best_hyperparams[0]}, Hidden Dimension={best_hyperparams[1]}, Regularization Lambda={best_hyperparams[2]}")

hyperparameter_tuning(X_train, y_train, X_val, y_val)
