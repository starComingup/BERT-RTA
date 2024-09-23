import dataset_utils as dru
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer, BertTokenizer, \
    BertForSequenceClassification, BertForMaskedLM
from bert_train_val import train, validate
import torch.optim.lr_scheduler as lr_scheduler
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

from torch_dataset.RedditDataset import RedditDataset
from torch_dataset.SpamDataset import SpamDataset
from torch_dataset.FinancialDataset import FinancialDataset
from torch_dataset.AmazonDataset import AmazonDataset
from torch_dataset.YoutubeDataset import YoutubeDataset
from torch_dataset.TwitterDataset import TwitterDataset
from torch_dataset.TwitterNDataset import TwitterNDataset
from torch_dataset.FakeJobDataset import FakeJobDataset

# 生成增强数据

def generate_augmented_data(model, device, tokenizer, texts, labels, num_augments=1, batch_size=64):
    model.eval()
    augmented_texts = []
    dataset = list(zip(texts, labels))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    with torch.no_grad():
        for batch in dataloader:
            batch_texts, batch_labels = batch
            # 确保 batch_texts 是字符串列表
            batch_texts = [str(text) for text in batch_texts]
            inputs = tokenizer.batch_encode_plus(batch_texts, return_tensors='pt', padding=True, truncation=True, max_length=128).to(device)
            outputs = model.generate(**inputs, max_length=128, num_return_sequences=num_augments)

            for i in range(len(batch_texts)):
                for j in range(num_augments):
                    augmented_text = tokenizer.decode(outputs[i * num_augments + j], skip_special_tokens=True)
                    augmented_texts.append((augmented_text, batch_labels[i]))

    return augmented_texts

num_classes = 3
aug_num = 1
result_file_path = 'result/cbert/youtube/result.csv'
confusion_pic_save_path = 'result/cbert/youtube/result_confusion_matrix.png'

model_path_name = "./dataroot/models/bert-base-uncased"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device1 = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 使用 BertForMaskedLM 模型
tokenizer = BertTokenizer.from_pretrained(model_path_name, padding_side='left')
bert_model = BertForMaskedLM.from_pretrained(model_path_name)
bert_model.to(device)
max_len = 64

finetune_epoch = 3
num_epochs = 30
batch_size = 128
lr = 3e-5
min_lr = 1e-8
T0 = 10
T_min = 1
alpha = 0.95

# dataset = RedditDataset(tokenizer=tokenizer, max_length=max_len)
# trainDataset, validDataset = dru.load_dataset(dataset, split_ratio=0.8)
# warn: modify file_path in BalancedDataset class
dataset = YoutubeDataset(tokenizer=tokenizer, max_length=max_len)
trainDataset, validDataset = dru.load_dataset(dataset, split_ratio=0.8)

train_dataloader = DataLoader(trainDataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(validDataset, batch_size=batch_size, shuffle=False)

finetune_optimizer = optim.AdamW(bert_model.parameters(), lr=1e-5)
bert_model.train()

# finetune the model
for num_epoch in range(finetune_epoch):
    total_loss = 0.0
    correct_predictions = 0
    for train_batch in train_dataloader:
        ids = train_batch['input_ids'].to(device)
        mask = train_batch['attention_mask'].to(device)
        train_labels = train_batch['labels'].to(device)
        train_outputs = bert_model(
            input_ids=ids,
            attention_mask=mask,
            labels=ids
        )
        loss = train_outputs.loss

        loss.backward()
        finetune_optimizer.step()
        finetune_optimizer.zero_grad()

        total_loss += loss.item()

    average_loss = total_loss / len(train_dataloader)
    print(f"Epoch {num_epoch + 1}, Loss: {average_loss:.4f}")

# aug data
augmented_data = generate_augmented_data(bert_model, device, tokenizer, trainDataset.texts, trainDataset.labels, aug_num)
# 将增强数据加入训练集
augmented_texts, augmented_labels = zip(*augmented_data)

trainDataset.append_data(augmented_texts, augmented_labels)
print(f'data augment successful!{len(trainDataset.labels)}')
torch.cuda.empty_cache()  # 清空缓存

batch_size = 128
bert_model = BertForSequenceClassification.from_pretrained(model_path_name, num_labels=num_classes)
bert_model.to(device1)
train_dataloader = DataLoader(trainDataset, batch_size=batch_size, shuffle=True)
optimizer = optim.AdamW(bert_model.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()
scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=T0, T_mult=1, eta_min=min_lr, last_epoch=-1)

epoch_list = []
train_loss_list = []
train_acc_list = []
train_recall_list = []
train_f1_list = []
valid_loss_list = []
valid_acc_list = []
valid_recall_list = []
valid_f1_list = []

# judge is multi classes or not
multi_classes = num_classes > 2
for epoch in range(num_epochs):
    train_loss, train_acc, train_recall, train_f1 = \
        train(bert_model, device1, train_dataloader, optimizer, criterion, multi_classes)
    val_loss, val_acc, val_recall, val_f1, conf_mat = validate(bert_model, device1, val_dataloader, criterion, multi_classes)

    if epoch == num_epochs - 1:  # 检查是否为最后一轮验证
        print(conf_mat)
        # 可视化混淆矩阵
        sns.heatmap(conf_mat, annot=True, cmap='Blues', fmt='d',
                    xticklabels=dataset.class_name, yticklabels=dataset.class_name)
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        plt.title('Confusion Matrix')
        # 设置图形质量参数（例如 DPI）
        plt.savefig(confusion_pic_save_path, dpi=300)  # 将图片保存为 'confusion_matrix.png'，300 DPI 表示较高分辨率
        plt.show()
    scheduler.step(epoch)

    print(f"Epoch {epoch + 1}/{num_epochs}")
    print(
        f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, Recall: {train_recall:.4f}, F1-score: {train_f1:.4f}")
    print(f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, Recall: {val_recall:.4f}, F1-score: {val_f1:.4f}")
    epoch_list.append(epoch + 1)
    train_acc_list.append(round(train_acc, 4))
    train_loss_list.append(round(train_loss, 4))
    train_recall_list.append(round(train_recall, 4))
    train_f1_list.append(round(train_f1, 4))
    valid_acc_list.append(round(val_acc, 4))
    valid_loss_list.append(round(val_loss, 4))
    valid_recall_list.append(round(val_recall, 4))
    valid_f1_list.append(round(val_f1, 4))

# create a blank file to store the index of DataFrame
columns = ['Epoch', 'Train Loss', 'Train Acc', 'Train Recall', 'Train F1', 'Val Loss', 'Val Acc', 'Val Recall',
           'Val F1']
df = pd.DataFrame(list(zip(epoch_list, train_loss_list, train_acc_list, train_recall_list, train_f1_list,
                           valid_loss_list, valid_acc_list, valid_recall_list, valid_f1_list)), columns=columns)
# save DataFrame to csv file
df.to_csv(result_file_path, index=False)