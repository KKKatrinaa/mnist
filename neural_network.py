"""
使用numpy搭建三层神经网络分类器，
能够自定义隐藏层大小、激活函数类型，支持通过反向传播计算给定损失的梯度。
"""

import numpy as np

class ThreeLayerNN:
    """三层神经网络分类器"""

    def __init__(self, input_dim, hidden_dim, output_dim, activation='relu', reg_lambda=0.01):
        self.params = {
            "W1": np.random.randn(hidden_dim, input_dim) * 0.01,
            "b1": np.zeros((hidden_dim, 1)),
            "W2": np.random.randn(output_dim, hidden_dim) * 0.01,
            "b2": np.zeros((output_dim, 1))
        }
        self.activation_name = activation
        self.reg_lambda = reg_lambda

    # 激活函数
    def activation(self, z):
        if self.activation_name == 'relu':
            return np.maximum(0, z)
        elif self.activation_name == 'sigmoid':
            return 1 / (1 + np.exp(-z))
        elif self.activation_name == 'tanh':
            return np.tanh(z)
        else:
            raise ValueError("Unsupported activation function")

    def activation_derivative(self, a):
        if self.activation_name == 'relu':
            return (a > 0).astype(float)
        elif self.activation_name == 'sigmoid':
            return a * (1 - a)
        elif self.activation_name == 'tanh':
            return 1 - np.power(a, 2)
        else:
            raise ValueError("Unsupported activation function")

    def softmax(self, z):
        e_z = np.exp(z - np.max(z, axis=0, keepdims=True))
        return e_z / np.sum(e_z, axis=0, keepdims=True)

    # 前向传播
    def forward(self, X):
        W1, b1, W2, b2 = self.params['W1'], self.params['b1'], self.params['W2'], self.params['b2']
        Z1 = np.dot(W1, X) + b1
        A1 = self.activation(Z1)
        Z2 = np.dot(W2, A1) + b2
        A2 = self.softmax(Z2)
        cache = (Z1, A1, Z2, A2)
        return A2, cache

    # 计算损失
    def compute_loss(self, Y_hat, Y):
        if Y.ndim == 2 and Y.shape[0] == Y_hat.shape[0]:  
            Y = np.argmax(Y, axis=0)  
        m = Y.shape[0]
        log_probs = -np.log(Y_hat[Y, np.arange(m)])  # 获取正确类别的对数概率
        loss = np.sum(log_probs) / m
        # 添加L2正则化
        loss += self.reg_lambda / 2 * (np.sum(np.square(self.params['W1'])) + np.sum(np.square(self.params['W2'])))
        return loss

    # 反向传播
    def backward(self, X, Y, cache):
        Z1, A1, Z2, A2 = cache
        m = X.shape[1]

        dZ2 = A2 - Y
        dW2 = np.dot(dZ2, A1.T) / m + self.reg_lambda * self.params['W2']
        db2 = np.sum(dZ2, axis=1, keepdims=True) / m

        dA1 = np.dot(self.params['W2'].T, dZ2)
        dZ1 = dA1 * self.activation_derivative(A1)
        dW1 = np.dot(dZ1, X.T) / m + self.reg_lambda * self.params['W1']
        db1 = np.sum(dZ1, axis=1, keepdims=True) / m

        grads = {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}
        return grads

    # 参数更新
    def update_params(self, grads, learning_rate):
        for key in ['W1', 'b1', 'W2', 'b2']:
            self.params[key] -= learning_rate * grads['d' + key]

