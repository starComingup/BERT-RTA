import torch
import numpy as np
import torch.nn as nn

import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from gensim.models import Word2Vec
import spacy


SEED = 1234
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

# 加载spaCy的英语模型
nlp = spacy.load('./dataroot/models/en_core_web_sm/en_core_web_sm/en_core_web_sm-3.7.1')

# 读取数据
file_path = './dataset/fake job posting/fake_job_postings.csv'  # 替换为你的数据集路径
result_file_path = './result/w2v+textcnn/fakejob/imbalanced.csv'
confusion_pic_save_path ='result/w2v+textcnn/fakejob/imbalanced.png'
# text_col_name = 'balanced_texts'
# label_col_name = 'balanced_labels'
subset = ['description','fraudulent']
content = pd.read_csv(file_path, usecols=subset)


text_col_name = 'description'
label_col_name = 'fraudulent'
# text_col_name = 'clean_text'
# label_col_name = 'category'
#
#
# proportion_to_keep = 0.1
# # # 对每个类别进行层次抽样以保持原分布
# sampled_data = []
# for category, group in content.groupby('category'):
#     category_sample = group.sample(frac=proportion_to_keep, random_state=1)
#     sampled_data.append(category_sample)
#
# content_sampled = pd.concat(sampled_data)
# content_sampled.reset_index(drop=True, inplace=True)
# content = content_sampled

# text_col_name = 'clean_comment'
# label_col_name = 'category'
# text_col_name = 'Sentence'
# label_col_name = 'Sentiment'
# content_filter = content[content['Sentiment'] != 'neutral'].copy()  # 使用 .copy() 创建副本
# content = content_filter.dropna(axis=0, how="any")  # 剔除标签为空白的样本

# text_col_name = 'content'
# label_col_name = 'sentiment'
# content_filter = content[(content['sentiment'] != 'worry')
#                                  & (content['sentiment'] != 'neutral') & (content['sentiment'] != 'empty')
#                                  & (content['sentiment'] != 'anger') & (content['sentiment'] != 'boredom')
#                                  & (content['sentiment'] != 'enthusiasm') & (content['sentiment'] != 'sadness')
#                                  & (content['sentiment'] != 'happiness')].copy()
# content_filter.reset_index(drop=True, inplace=True)
# content = content_filter

# content = pd.read_csv(file_path)
# content_filter = content[content['Sentiment'] != 1.0].copy()  # 使用 .copy() 创建副本
# content_filter.loc[content_filter['Sentiment'] == 2.0, 'Sentiment'] = 1.0
# content = content_filter.dropna(axis=0, how="any")  # 剔除标签为空白的样本
# text_col_name = 'Comment'
# label_col_name = 'Sentiment'

# text_col_name = 'Column2'
# label_col_name = 'Column1'
# content = pd.read_csv(file_path)

# text_col_name = 'reviewText'
# label_col_name = 'overall'
# content = pd.read_csv(file_path)
# content_filter = content[(content['overall'] != 3.0) & (content['overall'] != 4.0)].copy()
# content_filter.loc[content_filter[label_col_name] == 2.0, label_col_name] = 1.0
# content_filter.loc[content_filter[label_col_name] == 5.0, label_col_name] = 0.0
# content = content_filter.dropna(axis=0, how="any")  # 剔除标签为空白的样本

# text_col_name = 'clean_text'
# label_col_name = 'category'
# content_filter = content[content['category'] != 0].copy()  # 使用 .copy() 创建副本
# content_filter.loc[content_filter['category'] == 0, 'category'] = 0
# content = content_filter
# proportion_to_keep = 0.1
# # 对每个类别进行层次抽样以保持原分布
# sampled_data = []
# for category, group in content.groupby('category'):
#     category_sample = group.sample(frac=proportion_to_keep, random_state=1)
#     sampled_data.append(category_sample)
#
# content_sampled = pd.concat(sampled_data)
# content_sampled.reset_index(drop=True, inplace=True)
# content = content_sampled
# content_filter = content[content[label_col_name] != 1.0].copy()  # 使用 .copy() 创建副本
# content_filter.loc[content_filter[label_col_name] == 2.0, label_col_name] = 1.0
# df = content_filter.dropna(axis=0, how="any")
# df.reset_index(drop=True, inplace=True)
# content_filter = content[content[label_col_name] != 'neutral'].copy()
# content = content_filter

# df = content.dropna(subset=subset)  # 剔除标签为空白的样本
# df = df.reset_index(drop=True, inplace=True)  # 重置索引
df = content
# 划分训练集和测试集
train_df, test_df = train_test_split(df, test_size=0.2, random_state=SEED)
print(len(test_df))
# 分词和预处理文本
def tokenize_text(text):
    return [token.text for token in nlp(str(text))]

# 将文本数据转换为列表形式的分词后的文本
tokenized_train = [tokenize_text(text) for text in train_df[text_col_name].astype(str)]
tokenized_test = [tokenize_text(text) for text in test_df[text_col_name].astype(str)]

# 训练Word2Vec模型
w2v_model = Word2Vec(sentences=tokenized_train, vector_size=300, window=5, min_count=1, workers=4)

# 获取词汇表和权重矩阵
vocab = w2v_model.wv.index_to_key
print('vocab build success!')
embedding_weights = torch.FloatTensor(np.array([w2v_model.wv[word] for word in vocab]))

# 设置一个固定的序列长度
MAX_SEQUENCE_LENGTH = 50
# 将文本转换为嵌入向量
def get_embeddings(text):
    tokens = tokenize_text(text)
    indices = [vocab.index(token) if token in vocab else 0 for token in tokens[:MAX_SEQUENCE_LENGTH]]

    # 对于较短的句子，用零向量填充至最大长度；对于较长的句子，只取前max_length个词的嵌入
    embeddings_list = [embedding_weights[i] if i in indices else torch.zeros(embedding_weights.size(1))
                       for i in range(MAX_SEQUENCE_LENGTH)]

    # 将词嵌入向量按顺序拼接成一个张量，形状为 (sequence_length, embed_dim)
    return torch.unsqueeze(torch.stack(embeddings_list), dim=0)

# 使用LabelEncoder对标签进行编码
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(train_df[label_col_name])
y_test = label_encoder.transform(test_df[label_col_name])

# 构建自定义数据集
class TextDataset(Dataset):
    def __init__(self, X, y):
        self.X = [get_embeddings(text) for text in X]
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# 创建训练集和测试集的数据集实例
X_train = train_df[text_col_name].tolist()
X_test = test_df[text_col_name].tolist()
train_dataset = TextDataset(X_train, y_train)
test_dataset = TextDataset(X_test, y_test)
print(len(X_test))

# 创建 DataLoader 实例
batch_size = 128
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 定义模型
class TextCNN(nn.Module):
    def __init__(self, embed_dim, kernel_sizes=[3, 4, 5], num_filters=128, hidden_dim=256,
                 output_dim=len(label_encoder.classes_), dropout=0.5):
        super(TextCNN, self).__init__()
        self.embed_dim = embed_dim
        self.num_filters = num_filters
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.dropout = nn.Dropout(dropout)

        # 卷积层与批量归一化层（可选）
        self.convs = nn.ModuleList([nn.Conv1d(embed_dim, num_filters, kernel_size) for kernel_size in kernel_sizes])
        # 添加全局最大池化层
        self.pool = nn.AdaptiveMaxPool1d(1)

        # 全连接层
        self.fc1 = nn.Linear(num_filters * len(kernel_sizes), hidden_dim)
        self.relu = nn.ReLU()
        self.out = nn.Linear(hidden_dim, output_dim)

    def forward(self, text):
        # 将文本嵌入向量转为适合卷积层输入的形式：(batch_size, embed_dim, sequence_length)
        embedded_text = text.permute(0, 2, 1)

        # 对每个卷积核应用卷积操作并做最大池化
        conved = [conv(embedded_text).squeeze(-1) for conv in self.convs]
        pooled = [self.pool(conv).squeeze(-1) for conv in conved]

        # 将不同长度的卷积特征拼接起来
        concatenated = torch.cat(pooled, dim=1)
        dropout_layer = self.dropout(concatenated)

        # 通过全连接层得到分类结果
        fc_out = self.fc1(dropout_layer)
        fc_out = self.relu(fc_out)
        output = self.out(fc_out)

        return output

# 初始化模型参数
EMBEDDING_DIM = 300  # 与Word2Vec模型的维度一致
HIDDEN_DIM = 256
KERNEL_SIZES = [3, 4, 5]  # 可以调整的不同大小的卷积核尺寸
NUM_FILTERS = 128
OUTPUT_DIM = len(label_encoder.classes_)  # 输出维度为类别数
DROPOUT = 0.5

model = TextCNN(EMBEDDING_DIM, KERNEL_SIZES, NUM_FILTERS, HIDDEN_DIM, OUTPUT_DIM, DROPOUT)

# 定义优化器和损失函数
optimizer = optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss()

# 将模型移至GPU（如果可用）
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)
model = model.to(device)
criterion = criterion.to(device)

# 训练模型
N_EPOCHS = 40

epoch_list = []
train_loss_list = []
train_acc_list = []
train_recall_list = []
train_f1_list = []

for epoch in range(N_EPOCHS):
    model.train()
    total_loss = 0.0
    all_predictions_train = []
    all_labels_train = []

    for inputs, labels in train_loader:
        optimizer.zero_grad()
        inputs, labels = inputs.to(device), labels.to(device)


        predictions = model(inputs.squeeze(1)).squeeze(1)
        loss = criterion(predictions, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        predictions_labels = torch.argmax(predictions, dim=1)
        all_predictions_train.extend(predictions_labels.detach().cpu().numpy())
        all_labels_train.extend(labels.detach().cpu().numpy())

    # 计算训练集上的评估指标
    train_accuracy = accuracy_score(np.array(all_labels_train), np.array(all_predictions_train))
    train_recall = recall_score(np.array(all_labels_train), np.array(all_predictions_train), average='weighted')
    train_f1 = f1_score(np.array(all_labels_train), np.array(all_predictions_train), average='weighted')

    epoch_list.append(epoch + 1)
    train_acc_list.append(round(train_accuracy, 4))
    train_loss_list.append(round(total_loss / len(train_loader), 4))
    train_recall_list.append(round(train_recall, 4))
    train_f1_list.append(round(train_f1, 4))

    print(f'Epoch {epoch + 1}/{N_EPOCHS} => '
          f'Training Loss: {total_loss / len(train_loader):.4f}, '
          f'Training Accuracy: {train_accuracy:.4f}, '
          f'Training Recall: {train_recall:.4f}, '
          f'Training F1 Score: {train_f1:.4f}')

# 在所有 epochs 完成后在测试集上进行一次评估
model.eval()
all_predictions = []
all_labels = []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        predictions = model(inputs.squeeze(1)).squeeze(1)
        predictions_labels = torch.argmax(predictions, dim=1)
        all_predictions.extend(predictions_labels.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# 计算测试集上的评估指标
accuracy = accuracy_score(np.array(all_labels), np.array(all_predictions))
recall = recall_score(np.array(all_labels), np.array(all_predictions), average='weighted')
f1 = f1_score(np.array(all_labels), np.array(all_predictions), average='weighted')

print(f'Final Evaluation => '
      f'Accuracy: {accuracy:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}')

# 计算混淆矩阵
conf_mat = confusion_matrix(np.array(all_labels), np.array(all_predictions))
print(conf_mat)
# 可视化混淆矩阵
sns.heatmap(conf_mat, annot=True, cmap='Blues', fmt='d',
            xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
# 设置图形质量参数（例如 DPI）
plt.savefig(confusion_pic_save_path, dpi=300)  # 将图片保存为 'confusion_matrix.png'，300 DPI 表示较高分辨率
plt.show()
# create a blank file to store the index of DataFrame
columns = ['Epoch', 'Train Loss', 'Train Acc', 'Train Recall', 'Train F1']
df = pd.DataFrame(list(zip(epoch_list, train_loss_list, train_acc_list
                           , train_recall_list, train_f1_list)), columns=columns)
df.to_csv(result_file_path, index=False)