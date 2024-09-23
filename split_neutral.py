import pandas as pd
import numpy as np
import torch
from transformers import BertTokenizer, BertModel
from torch.utils.data import DataLoader, Dataset
from sklearn.cluster import AgglomerativeClustering, DBSCAN
# 读取数据
file_path = './dataset/Amazon Musical Instruments Reviews/Musical_instruments_reviews.csv'  # 替换为你的数据集路径

content = pd.read_csv(file_path)

text_col_name = 'reviewText'
df = content.dropna(axis=0, how="any")  # 剔除标签为空白的样本
df.reset_index(drop=True, inplace=True)  # 重置索引

model_name = './dataroot/models/bert-base-uncased'
# 加载预训练的BERT模型和tokenizer
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)

class MyDataset(Dataset) :
    def __init__(self, data, text_col, token, max_length):
        self.text = data[text_col]
        self.tokenizer = token
        self.max_length = max_length

    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):
        text = str(self.text[index])

        inputs = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        input_ids = inputs['input_ids'].squeeze()
        attention_mask = inputs['attention_mask'].squeeze()
        return input_ids, attention_mask


batch_size = 128
max_len = 128
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)
dataset = MyDataset(df, text_col_name, tokenizer, max_len)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
sentence_embeddings = []

with torch.no_grad():
    for batch in dataloader:
        ids, mask = batch
        ids, mask = ids.to(device), mask.to(device)
        outputs = model(ids, attention_mask=mask)
        sentence_embeddings.extend(outputs[1].cpu().numpy())



# 如果 sentence_embeddings 中的元素不是 numpy 数组，先将它们转换为 numpy 数组
sentence_embeddings = [np.array(embedding) for embedding in sentence_embeddings]

# 将列表转换为 numpy 数组
sentence_embeddings = np.array(sentence_embeddings)
n_clusters = 3
affinity = 'cosine'
linkage = 'average'
# n_clusters 是您希望的聚类数量
# affinity 是相似度或距离度量的选择，如 'euclidean' 或 'cosine'
# linkage 是链接准则的选择，如 'ward'、'complete' 或 'average'
clustering = AgglomerativeClustering(n_clusters=n_clusters, affinity=affinity, linkage=linkage)
cluster_labels = clustering.fit_predict(sentence_embeddings)

cluster_i_sentences = []
for cluster_id in range(n_clusters):
    cluster_sentences = [sentence for sentence, label in zip(dataset.text, cluster_labels) if label == cluster_id]
    cluster_i_sentences.append(cluster_sentences[:10])
print(cluster_i_sentences)
# 定义 DBSCAN 的参数，如 eps 和 min_samples
# eps = 5  # 邻域半径
# min_samples = 150  # 最小样本数
# dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean')
# cluster_labels = dbscan.fit_predict(sentence_embeddings)

from sklearn.preprocessing import MinMaxScaler

# 使用 MinMaxScaler 进行归一化处理
scaler = MinMaxScaler()
sentence_embeddings_normalized = scaler.fit_transform(sentence_embeddings)
sentence_embeddings = sentence_embeddings_normalized

# from sklearn.decomposition import LatentDirichletAllocation
# import matplotlib.pyplot as plt
# # 使用 LDA 进行降维
# n_topics = 2  # 指定主题数
# lda = LatentDirichletAllocation(n_components=n_topics, random_state=0)
# X_lda = lda.fit_transform(sentence_embeddings)

# 绘制三维散点图
from mpl_toolkits.mplot3d import Axes3D
# fig = plt.figure(figsize=(12, 8))
# ax = fig.add_subplot(111, projection='3d')
# # scatter = ax.scatter(X_tsne[:, 0], X_tsne[:, 1], X_tsne[:, 2], c=cluster_labels, cmap='viridis')
# scatter = ax.scatter(X_lda[:, 0], X_lda[:, 1], X_lda[:, 2], c=cluster_labels, cmap='viridis')
# plt.colorbar(scatter)
# plt.title('Hierarchical Clustering Visualization (3D)')
# plt.show()

# # 使用 t-SNE 进行降维
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
tsne = TSNE(n_components=2, random_state=0)
X_tsne = tsne.fit_transform(sentence_embeddings)
#
# # 绘制散点图
plt.figure(figsize=(10, 6))
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=cluster_labels, cmap='viridis')
plt.title('Hierarchical Clustering Visualization')
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.colorbar()
plt.show()
