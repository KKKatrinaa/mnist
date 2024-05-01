# mnist
构建三层神经网络分类器，实现图像分类
## 模型训练
运行'save_model.py'可以进行模型训练，并保存模型权重
## 模型测试
运行'test_model.py'可以进行模型测试，并输出测试结果准确率
## 文件说明
'preprocess_data.py'：加载数据并预处理。
'neural_network.py'：构建三层神经网络分类器。
'train_model.py'：模型训练和评估函数。
'search_params.py'：超参数查找。
'save_model.py'：调用最优参数训练模型，保存模型权重和训练集验证集损失和准确率。
'test_model.py'：模型测试，输出测试结果准确率。
'plot.py'：对训练集和验证集损失和准确率可视化，模型网络参数可视化。
'model_weights.npz'：模型权重。
'model_history.json'：模型训练历史，包括训练集损失，验证集损失和准确率。
'data'文件夹：原始数据，包含训练集图像和标签，测试集图像和标签。
'visual_result'文件夹：图形可视化结果。