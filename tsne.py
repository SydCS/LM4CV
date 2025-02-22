import torch
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# 假设model[0].weight.data和attributes_embeddings都是随机生成的数据
n = 200  # 数据点数量

# 从文件加载数据
loaded_data = torch.load("weights_and_embeddings.pth")

# 从字典中提取权重和嵌入
model_weights = loaded_data["model_weights"]
attributes_embeddings = loaded_data["attributes_embeddings"]

# 合并数据
data = np.vstack((model_weights, attributes_embeddings))

# 应用t-SNE
tsne = TSNE(n_components=2, random_state=42)
data_2d = tsne.fit_transform(data)

# 可视化
plt.figure(figsize=(10, 5))
plt.scatter(data_2d[:n, 0], data_2d[:n, 1], c="blue", label="Model Weights")
plt.scatter(data_2d[n:, 0], data_2d[n:, 1], c="red", label="Attributes Embeddings")
plt.legend()
plt.title("t-SNE Visualization of Model Weights and Attributes Embeddings")
plt.xlabel("Component 1")
plt.ylabel("Component 2")

# 保存图像到文件
plt.savefig("tsne.png")
