import matplotlib.pyplot as plt
import json
import seaborn as sns
import numpy as np

# 权重热力图
def plot_heatmap(weights, layer_name, filepath=None):
    plt.figure(figsize=(10, 8))
    sns.heatmap(weights, annot=False, cmap='coolwarm', center=0)
    plt.title(f"Heatmap for {layer_name}")
    plt.xlabel("Input Neurons")
    plt.ylabel("Output Neurons")
    if filepath:
        plt.savefig(filepath)  # 保存图像
    plt.show()

# 权重直方图
def plot_histogram(weights, layer_name, filepath=None):
    plt.figure(figsize=(6, 4))
    plt.hist(weights.flatten(), bins=50, alpha=0.75)
    plt.title(f"Histogram of weights in {layer_name}")
    plt.xlabel("Weight value")
    plt.ylabel("Frequency")
    plt.grid(True)
    if filepath:
        plt.savefig(filepath)  # 保存图像
    plt.show()

# 偏移项直方图
def plot_biases(biases, layer_name, filepath=None):
    plt.figure(figsize=(6, 4))
    plt.bar(range(len(biases)), biases)
    plt.title(f"Biases in {layer_name}")
    plt.xlabel("Neuron")
    plt.ylabel("Bias value")
    if filepath:
        plt.savefig(filepath)  # 保存图像
    plt.show()

# 损失和准确率曲线
def plot_training_history(history, filepath=None):
    plt.figure(figsize=(14, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss', linestyle='--')
    plt.title('Loss during Training')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history['val_accuracy'], label='Validation Accuracy', color='red')
    plt.title('Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    if filepath:
        plt.savefig(filepath)

    plt.show()


with open('model_history.json', 'r') as f:
    history = json.load(f)

plot_training_history(history, "visual_result/loss_and_accuracy.png")

def load_weights(filepath='model_weights.npz'):
    data = np.load(filepath)
    W1 = data['W1']
    W2 = data['W2']
    b1 = data['b1'].flatten()  
    b2 = data['b2'].flatten()
    return W1, W2, b1, b2

W1, W2, b1, b2 = load_weights()

plot_heatmap(W1, "W1", "visual_result/heatmap_W1.png")
plot_heatmap(W2, "W2", "visual_result/heatmap_W2.png")
plot_histogram(W1, "W1", "visual_result/histogram_W1.png")
plot_histogram(W2, "W2", "visual_result/histogram_W2.png")
plot_biases(b1, "b1", "visual_result/biases_b1.png")
plot_biases(b2, "b2", "visual_result/biases_b2.png")
