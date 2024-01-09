import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from gensim.models import Word2Vec
import spacy


SEED = 1234
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

# 加载spaCy的英语模型
nlp = spacy.load('./dataroot/models/en_core_web_sm/en_core_web_sm/en_core_web_sm-3.7.1')

# 读取数据
file_path = './dataset/youtube statistic/comments.csv'  # 替换为你的数据集路径
result_file_path = './result/w2v+lstm/youtube_w2v_lstm_imbalance.csv'
text_col_name = 'Comment'
label_col_name = 'Sentiment'
df = pd.read_csv(file_path)
df = df.dropna(axis=0, how="any")
df.reset_index(drop=True, inplace=True)

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

# 创建 DataLoader 实例
batch_size = 128
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 定义模型
class BiLSTM(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, output_dim, dropout):
        super(BiLSTM, self).__init__()
        self.embedding_dim = embedding_dim
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, bidirectional=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text):
        outputs, (hidden, cell) = self.lstm(text.unsqueeze(0))
        hidden = self.dropout(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))
        return self.fc(hidden)

# 初始化模型参数
EMBEDDING_DIM = 300  # 与Word2Vec模型的维度一致
HIDDEN_DIM = 256
OUTPUT_DIM = len(label_encoder.classes_)  # 输出维度为类别数
DROPOUT = 0.5

model = BiLSTM(EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM, DROPOUT)

# 定义优化器和损失函数
optimizer = optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss()

# 将模型移至GPU（如果可用）
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)
model = model.to(device)
criterion = criterion.to(device)

# 训练模型
N_EPOCHS = 30

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

print(f'Final Evaluation => '
      f'Accuracy: {accuracy:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}')

# create a blank file to store the index of DataFrame
columns = ['Epoch', 'Train Loss', 'Train Acc', 'Train Recall', 'Train F1']
df = pd.DataFrame(list(zip(epoch_list, train_loss_list, train_acc_list
                           , train_recall_list, train_f1_list)), columns=columns)
df.to_csv(result_file_path, index=False)