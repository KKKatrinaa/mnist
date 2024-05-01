"""
训练和评估模型，实现SGD优化器、学习率下降、交叉熵损失和L2正则化，并能根据验证集指标自动保存最优的模型权重
"""

import numpy as np
from preprocess_data import *
from neural_network import *

def train(model, X_train, y_train, X_val, y_val, num_epochs=100, learning_rate=0.01, batch_size=32, lr_decay=0.99):
    # 训练模型
    best_val_acc = 0
    best_params = {}

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_accuracy": []
    }
    
    lr = learning_rate

    for epoch in range(num_epochs):
        permutation = np.random.permutation(X_train.shape[0])
        X_train_shuffled = X_train[permutation]
        y_train_shuffled = y_train[permutation]

        epoch_losses = []

        for i in range(0, X_train.shape[0], batch_size):
            X_batch = X_train_shuffled[i:i + batch_size].T
            y_batch = y_train_shuffled[i:i + batch_size]

            # 转换为独热编码
            y_one_hot = np.zeros((10, X_batch.shape[1]))
            y_one_hot[y_batch, np.arange(X_batch.shape[1])] = 1

            # 前向传播
            outputs, cache = model.forward(X_batch)
            loss = model.compute_loss(outputs, y_one_hot)
            epoch_losses.append(loss)

            # 反向传递
            grads = model.backward(X_batch, y_one_hot, cache)
            model.update_params(grads, lr)

        lr *= lr_decay

        average_loss = np.mean(epoch_losses)
        history['train_loss'].append(average_loss)

        # 在验证集上进行评估
        val_outputs, _ = model.forward(X_val.T)
        val_loss = model.compute_loss(val_outputs, y_val)
        val_acc = evaluate(model, X_val, y_val)

        history['val_accuracy'].append(val_acc)
        history['val_loss'].append(val_loss)

        print(f"Epoch {epoch+1}, Train Loss: {average_loss:.2f}, Validation Loss: {val_loss:.2f}, Validation Accuracy: {val_acc:.2f}")

        # 保存最优模型的参数
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_params = model.params.copy()
        
    model.params = best_params
    return history

def evaluate(model, X, y):
    # 模型预测评估
    outputs, _ = model.forward(X.T)
    predictions = np.argmax(outputs, axis=0)
    return np.mean(predictions == y)

