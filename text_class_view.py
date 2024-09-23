# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.manifold import TSNE
# import pandas as pd
# import torch
# from torch.utils.data import TensorDataset, DataLoader
# from transformers import BertModel, BertTokenizer
#
# from dataset_utils.mix_text_util import label_mapping
#
# # 载入BERT模型和分词器
# model_name = './dataroot/models/bert-base-uncased'
# model = BertModel.from_pretrained(model_name).to('cuda')
# tokenizer = BertTokenizer.from_pretrained(model_name)
#
# # 读取数据
# file_path = './dataset/spam/spam1.csv'  # 替换为你的数据集路径
#
# content = pd.read_csv(file_path)
#
# df = content.dropna(axis=0, how="any")  # 剔除标签为空白的样本
# df.reset_index(drop=True, inplace=True)  # 重置索引
#
# # 假设有一个文本列表和对应的标签
# texts = df['Column2'].tolist()
# labels = df['Column1'].tolist()  # 示例标签
# labels = label_mapping(labels)
#
# # 获取BERT嵌入
# def get_bert_embeddings(texts, batch_size=128):
#     inputs = tokenizer(texts, return_tensors='pt', max_length=128, truncation=True, padding='max_length')
#     dataset = TensorDataset(inputs['input_ids'], inputs['attention_mask'])
#     dataloader = DataLoader(dataset, batch_size=batch_size)
#
#     embeddings = []
#     model.eval()
#     with torch.no_grad():
#         for batch in dataloader:
#             input_ids, attention_mask = [b.to('cuda') for b in batch]
#             outputs = model(input_ids=input_ids, attention_mask=attention_mask)
#             embeddings.append(outputs.last_hidden_state[:, 0, :].cpu().numpy())  # 使用CLS token的嵌入
#     return np.concatenate(embeddings, axis=0)
#
# embeddings = get_bert_embeddings(texts)
#
# # t-SNE降维
# tsne = TSNE(n_components=2, random_state=42)
# embeddings_2d = tsne.fit_transform(embeddings)
#
# # 2D散点图可视化
# plt.figure(figsize=(10, 8))
# scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=labels, cmap='viridis')
#
# # 添加图例
# legend1 = plt.legend(*scatter.legend_elements(), title="Classes")
# plt.gca().add_artist(legend1)
#
# plt.xlabel('t-SNE Dimension 1')
# plt.ylabel('t-SNE Dimension 2')
# plt.title('t-SNE Visualization of BERT Embeddings with Text Labels')
# plt.show()

# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.manifold import TSNE
# import pandas as pd
# import torch
# from torch.utils.data import TensorDataset, DataLoader
# from transformers import BertModel, BertTokenizer, BertForSequenceClassification
#
# from dataset_utils.mix_text_util import label_mapping
#
# # 载入BERT模型和分词器
# model_name = './dataroot/models/bert-base-uncased'
# model = BertModel.from_pretrained(model_name).to('cuda')
# tokenizer = BertTokenizer.from_pretrained(model_name)
#
# # 读取数据
# file_path = './dataset/spam/spam1.csv'  # 替换为你的数据集路径
#
# content = pd.read_csv(file_path)
#
# df = content.dropna(axis=0, how="any")  # 剔除标签为空白的样本
# df.reset_index(drop=True, inplace=True)  # 重置索引
#
# # 假设有一个文本列表和对应的标签
# texts = df['Column2'].tolist()
# labels = df['Column1'].tolist()  # 示例标签
# labels = label_mapping(labels)
#
# # 获取BERT嵌入
# def get_bert_embeddings(texts, batch_size=128):
#     inputs = tokenizer(texts, return_tensors='pt', max_length=128, truncation=True, padding='max_length')
#     dataset = TensorDataset(inputs['input_ids'], inputs['attention_mask'])
#     dataloader = DataLoader(dataset, batch_size=batch_size)
#
#     embeddings = []
#     model.eval()
#     with torch.no_grad():
#         for batch in dataloader:
#             input_ids, attention_mask = [b.to('cuda') for b in batch]
#             outputs = model(input_ids=input_ids, attention_mask=attention_mask)
#             embeddings.append(outputs.last_hidden_state[:, 0, :].cpu().numpy())  # 使用CLS token的嵌入
#     return np.concatenate(embeddings, axis=0)
#
# embeddings = get_bert_embeddings(texts)
#
# # 获取预测结果
# def get_predictions(texts, bert_model, batch_size=128):
#     inputs = tokenizer(texts, return_tensors='pt', max_length=128, truncation=True, padding='max_length')
#     dataset = TensorDataset(inputs['input_ids'], inputs['attention_mask'])
#     dataloader = DataLoader(dataset, batch_size=batch_size)
#
#     predictions = []
#     model.eval()
#     with torch.no_grad():
#         for batch in dataloader:
#             input_ids, attention_mask = [b.to('cuda') for b in batch]
#             outputs = bert_model(input_ids=input_ids, attention_mask=attention_mask)
#             logits = outputs.logits
#             preds = torch.argmax(logits, dim=1).cpu().numpy()
#             predictions.extend(preds)
#     return predictions
#
# # 这里假设有一个分类模型 `classifier` 并对其进行预测
# bert_model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2).to('cuda')
# predictions = get_predictions(texts, bert_model)
#
# # 标识分类错误的样本
# errors = [1 if pred != true else 0 for pred, true in zip(predictions, labels)]
#
# # t-SNE降维
# tsne = TSNE(n_components=2, random_state=42)
# embeddings_2d = tsne.fit_transform(embeddings)
#
# # 2D散点图可视化
# plt.figure(figsize=(12, 10))
#
# # 分类正确和错误的点分别标记
# for label in np.unique(labels):
#     correct_indices = [i for i, e in enumerate(errors) if e == 0 and labels[i] == label]
#     error_indices = [i for i, e in enumerate(errors) if e == 1 and labels[i] == label]
#
#     correct_embeddings = embeddings_2d[correct_indices]
#     error_embeddings = embeddings_2d[error_indices]
#
#     plt.scatter(correct_embeddings[:, 0], correct_embeddings[:, 1], label=f'Class {label} - Correct', alpha=0.6)
#     plt.scatter(error_embeddings[:, 0], error_embeddings[:, 1], label=f'Class {label} - Error', marker='x')
#
# # 添加图例
# plt.legend()
# plt.xlabel('t-SNE Dimension 1')
# plt.ylabel('t-SNE Dimension 2')
# plt.title('t-SNE Visualization of BERT Embeddings with Classification Errors')
# plt.show()


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

# 读取数据
file_path = './dataset/spam/spam1.csv'  # 替换为你的数据集路径
content = pd.read_csv(file_path)
df = content.dropna(axis=0, how="any")  # 剔除标签为空白的样本
df.reset_index(drop=True, inplace=True)  # 重置索引

# 假设有一个文本列表和对应的标签
texts = df['Column2'].tolist()
labels = df['Column1'].tolist()  # 示例标签
labels = label_mapping(labels)


# 创建数据集和数据加载器
def create_dataloader(texts, labels, batch_size=128, sampler_type=RandomSampler):
    inputs = tokenizer(texts, return_tensors='pt', max_length=128, truncation=True, padding='max_length')
    labels = torch.tensor(labels)
    dataset = TensorDataset(inputs['input_ids'], inputs['attention_mask'], labels)
    sampler = sampler_type(dataset)
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=batch_size)
    return dataloader


train_dataloader = create_dataloader(texts, labels, batch_size=128, sampler_type=RandomSampler)


# 微调模型
def fine_tune_model(model, train_dataloader, epochs=1, learning_rate=2e-5):
    optimizer = AdamW(model.parameters(), lr=learning_rate, eps=1e-8)
    total_steps = len(train_dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    loss_fn = torch.nn.CrossEntropyLoss()

    for epoch in range(epochs):
        # 训练阶段
        model.train()
        total_loss = 0
        for batch in train_dataloader:
            b_input_ids, b_input_mask, b_labels = [item.to('cuda') for item in batch]
            model.zero_grad()
            outputs = model(input_ids=b_input_ids, attention_mask=b_input_mask, labels=b_labels)
            loss = outputs.loss
            total_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

        avg_train_loss = total_loss / len(train_dataloader)
        print(f'Epoch {epoch + 1}/{epochs}, Training loss: {avg_train_loss}')


fine_tune_model(model, train_dataloader, epochs=3, learning_rate=2e-5)


# 获取BERT嵌入
def get_bert_embeddings(texts, batch_size=128):
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


# 获取预测结果
def get_predictions(texts, batch_size=128):
    inputs = tokenizer(texts, return_tensors='pt', max_length=128, truncation=True, padding='max_length')
    dataset = TensorDataset(inputs['input_ids'], inputs['attention_mask'])
    dataloader = DataLoader(dataset, batch_size=batch_size)

    predictions = []
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            input_ids, attention_mask = [b.to('cuda') for b in batch]
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            predictions.extend(preds)
    return predictions


# 进行微调后的嵌入和预测
embeddings = get_bert_embeddings(texts)
predictions = get_predictions(texts)

# 标识分类错误的样本
errors = [1 if pred != true else 0 for pred, true in zip(predictions, labels)]

# t-SNE降维
tsne = TSNE(n_components=2, random_state=42)
embeddings_2d = tsne.fit_transform(embeddings)

# 2D散点图可视化
plt.figure(figsize=(12, 10))

# 分类正确和错误的点分别标记
for label in np.unique(labels):
    correct_indices = [i for i, e in enumerate(errors) if e == 0 and labels[i] == label]
    error_indices = [i for i, e in enumerate(errors) if e == 1 and labels[i] == label]

    correct_embeddings = embeddings_2d[correct_indices]
    error_embeddings = embeddings_2d[error_indices]

    plt.scatter(correct_embeddings[:, 0], correct_embeddings[:, 1], label=f'Class {label} - Correct', alpha=0.6)
    plt.scatter(error_embeddings[:, 0], error_embeddings[:, 1], label=f'Class {label} - Error', marker='x')

# 添加图例
plt.legend()
plt.xlabel('t-SNE Dimension 1')
plt.ylabel('t-SNE Dimension 2')
plt.title('t-SNE Visualization of BERT Embeddings with Classification Errors after 3 epochs Fine-Tuning')
plt.show()
