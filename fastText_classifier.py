import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from gensim.models import Word2Vec
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import spacy


SEED = 1234
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

# 加载spaCy的英语模型
nlp = spacy.load('./dataroot/models/en_core_web_sm/en_core_web_sm/en_core_web_sm-3.7.1')

# 读取数据
file_path = './dataset/financial sentiment analysis/data.csv'  # 替换为你的数据集路径
result_file_path = 'result/w2v+classifier/financial_w2v_imbalance.csv'
confusion_pic_save_path ='result/w2v+classifier/financial_imbalance_confusion_matrix.png'

# text_col_name = 'balanced_texts'
# label_col_name = 'balanced_labels'
content = pd.read_csv(file_path)

text_col_name = 'Sentence'
label_col_name = 'Sentiment'
content_filter = content[content['Sentiment'] != 'neutral'].copy()  # 使用 .copy() 创建副本
content = content_filter.dropna(axis=0, how="any")  # 剔除标签为空白的样本

# text_col_name = 'content'
# label_col_name = 'sentiment'
# content_filter = content[(content['sentiment'] != 'worry')
#                                  & (content['sentiment'] != 'neutral') & (content['sentiment'] != 'empty')
#                                  & (content['sentiment'] != 'anger') & (content['sentiment'] != 'boredom')
#                                  & (content['sentiment'] != 'enthusiasm') & (content['sentiment'] != 'sadness')
#                                  & (content['sentiment'] != 'happiness')].copy()
# content_filter.reset_index(drop=True, inplace=True)
# content = content_filter

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

# content = pd.read_csv(file_path)
# content_filter = content[content['Sentiment'] != 1.0].copy()  # 使用 .copy() 创建副本
# content_filter.loc[content_filter['Sentiment'] == 2.0, 'Sentiment'] = 1.0
# content = content_filter.dropna(axis=0, how="any")  # 剔除标签为空白的样本
# text_col_name = 'Comment'
# label_col_name = 'Sentiment'

# text_col_name = 'reviewText'
# label_col_name = 'overall'
# content = pd.read_csv(file_path)
# content_filter = content[(content['overall'] != 3.0) & (content['overall'] != 4.0)].copy()
# content_filter.loc[content_filter[label_col_name] == 2.0, label_col_name] = 1.0
# content_filter.loc[content_filter[label_col_name] == 5.0, label_col_name] = 0.0
# df = content_filter.dropna(axis=0, how="any")  # 剔除标签为空白的样本

# text_col_name = 'Column2'
# label_col_name = 'Column1'
# content = pd.read_csv(file_path)

df = content.dropna(axis=0, how="any")  # 剔除标签为空白的样本
df.reset_index(drop=True, inplace=True)  # 重置索引
# 划分训练集和测试集
train_df, test_df = train_test_split(df, test_size=0.2, random_state=SEED)

# 分词和预处理文本
def tokenize_text(text):
    return [token.text for token in nlp(text)]

# 将文本数据转换为列表形式的分词后的文本
tokenized_train = [tokenize_text(text) for text in train_df[text_col_name]]
tokenized_test = [tokenize_text(text) for text in test_df[text_col_name]]

# 训练Word2Vec模型
w2v_model = Word2Vec(sentences=tokenized_train, vector_size=300, window=5, min_count=1, workers=4)

# 获取词汇表和权重矩阵
vocab = w2v_model.wv.index_to_key
print('vocab build success!')
embedding_weights = torch.FloatTensor(np.array([w2v_model.wv[word] for word in vocab]))

# 将文本转换为嵌入向量
def get_embeddings(text):
    tokens = tokenize_text(text)
    indices = [vocab.index(token) for token in tokens if token in vocab]
    if not indices:
        # 如果文本中没有在词汇表中的单词，则返回全零向量
        return torch.zeros(embedding_weights.size(1))
    return embedding_weights[indices].mean(dim=0)

# 使用LabelEncoder对标签进行编码
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(train_df[label_col_name])
y_test = label_encoder.transform(test_df[label_col_name])

# 构建自定义数据集00
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

# 创建 DataLoader 实例
batch_size = 512
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 定义模型
class TextClassifier(nn.Module):
    def __init__(self, embed_dim, hidden_dim, output_dim):
        super(TextClassifier, self).__init__()
        self.fc = nn.Linear(embed_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(DROPOUT)
        self.out = nn.Linear(hidden_dim, output_dim)

    def forward(self, text):
        hidden = self.fc(text)
        hidden = self.relu(hidden)
        hidden = self.dropout(hidden)
        output = self.out(hidden)
        return output

# 初始化模型参数
EMBEDDING_DIM = 300  # 与Fasttext模型的维度一致
HIDDEN_DIM = 256
OUTPUT_DIM = len(label_encoder.classes_)  # 输出维度为类别数
DROPOUT = 0.5


model = TextClassifier(EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM)

# 定义优化器和损失函数
optimizer = optim.SGD(model.parameters(), lr=0.01)
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
        predictions = model(inputs).squeeze(1)
        loss = criterion(predictions, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        predictions_labels = torch.argmax(predictions, dim=1)
        all_predictions_train.extend(predictions_labels.detach().cpu().numpy())
        all_labels_train.extend(labels.detach().cpu().numpy())

    # print(all_predictions_train)
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
        predictions = model(inputs).squeeze(1)
        predictions_labels = torch.argmax(predictions, dim=1)
        all_predictions.extend(predictions_labels.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# 计算测试集上的评估指标
accuracy = accuracy_score(np.array(all_labels), np.array(all_predictions))
recall = recall_score(np.array(all_labels), np.array(all_predictions), average='weighted')
f1 = f1_score(np.array(all_labels), np.array(all_predictions), average='weighted')

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

print(f'Final Evaluation => '
      f'Accuracy: {accuracy:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}')

# create a blank file to store the index of DataFrame
columns = ['Epoch', 'Train Loss', 'Train Acc', 'Train Recall', 'Train F1']
df = pd.DataFrame(list(zip(epoch_list, train_loss_list, train_acc_list
                           , train_recall_list, train_f1_list)), columns=columns)
df.to_csv(result_file_path, index=False)