import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup

from dataset_utils.mix_text_util import label_mapping

# 载入BERT模型和分词器
model_name = './dataroot/models/bert-base-uncased'
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2).to('cuda')
tokenizer = BertTokenizer.from_pretrained(model_name)

origin_data_file="./dataset/fake job posting/fake_job_postings.csv"
aug_data_file="./dataset/fake job posting/eda_selected.csv"
fig_path="./result/cluster/spam-nocluster.png"
fig_title='Spam: t-SNE Results with BERT Embeddings'
origin_data_text_col="description"
aug_data_text_col="balanced_texts"
origin_data_label_col="fraudulent"
aug_data_label_col="balanced_labels"

# 读取数据
file_path = origin_data_file  # 替换为你的数据集路径
content0 = pd.read_csv(file_path)
df0 = content0.dropna(axis=0, how="any")  # 剔除标签为空白的样本
df0.reset_index(drop=True, inplace=True)  # 重置索引

# proportion_to_keep = 0.1
# # 对每个类别进行层次抽样以保持原分布
# sampled_data = []
# for category, group in df0.groupby('category'):
#     category_sample = group.sample(frac=proportion_to_keep, random_state=1)
#     sampled_data.append(category_sample)
#
# content_sampled = pd.concat(sampled_data)
# content_sampled.reset_index(drop=True, inplace=True)
# origin_texts = content_sampled['clean_text'].tolist()
# labels = content_sampled['category'].tolist()

origin_texts = df0[origin_data_text_col].tolist()
labels = df0[origin_data_label_col].tolist()  # 示例标签
origin_labels = label_mapping(labels)

# 读取数据
file_path = aug_data_file  # 替换为你的数据集路径
content0 = pd.read_csv(file_path)
df0 = content0.dropna(axis=0, how="any")  # 剔除标签为空白的样本
df0.reset_index(drop=True, inplace=True)  # 重置索引

aug_texts = df0[aug_data_text_col].tolist()
aug_labels = df0[aug_data_label_col].tolist()  # 示例标签

def get_embeddings(texts, batch_size=128):
    inputs = tokenizer(texts, return_tensors='pt', max_length=128, truncation=True, padding='max_length')
    dataset = TensorDataset(inputs['input_ids'], inputs['attention_mask'])
    dataloader = DataLoader(dataset, batch_size=batch_size)

    embeddings = []
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            input_ids, attention_mask = [b.to('cuda') for b in batch]
            outputs = model.bert(input_ids=input_ids, attention_mask=attention_mask)
            embeddings.append(outputs.last_hidden_state[:, 0, :].cpu().numpy())  # 使用CLS token的嵌入
    return np.concatenate(embeddings, axis=0)

origin_embedding = get_embeddings(origin_texts)
aug_embedding = get_embeddings(aug_texts)

# embeddings = np.concatenate((origin_embedding,aug_embedding),axis=0)
embeddings = origin_embedding
from skfuzzy.cluster import cmeans
from sklearn.cluster import KMeans

from sklearn.metrics import silhouette_score
num_clusters=2
# kmeans = KMeans(n_clusters=num_clusters,random_state=0)
# labels = kmeans.fit_predict(embeddings)
data = np.array(embeddings)
cmeans_params = (data.T, num_clusters, 2)
cntr, u, u0, d, jm, p, fpc = cmeans(*cmeans_params,error=0.005, maxiter=1000)
labels = np.argmax(u, axis=0)

score= silhouette_score(embeddings, labels)
print(score)
exit(1)

tsne = TSNE(n_components=2,random_state=0)
embeddings = tsne.fit_transform(embeddings)


origin_embedding = embeddings[:len(origin_labels)]
aug_embedding = embeddings[len(origin_labels):]

plt.figure(figsize=(10, 8))
colors={0:"green", 1:"blue", 2:"yellow"}
# 绘制第一个数据集
for label in np.unique(origin_labels):
    indices = (origin_labels == label)
    plt.scatter(origin_embedding[indices, 0], origin_embedding[indices, 1],
                c=colors[label], label=f'Original dataset - Cluster {label}')

# 绘制第二个数据集
for label in np.unique(aug_labels):
    indices = (aug_labels == label)
    plt.scatter(aug_embedding[indices, 0], aug_embedding[indices, 1],
        marker='x', c=colors[label], label=f'Augment dataset - Cluster {label}')
# # 用不同颜色区分原始数据集和增强数据集
# plt.scatter(origin_embedding[:,0], origin_embedding[:,1], c='blue', label='Original Data')
# plt.scatter(aug_embedding[:, 0], aug_embedding[:, 1], c='green', label='Augmented Data')

plt.savefig(fig_path, dpi=300)
plt.title(fig_title)
plt.legend()
plt.show()

