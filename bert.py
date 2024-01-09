import dataset_utils as dru
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer, BertTokenizer, \
    BertForSequenceClassification
from sklearn.metrics import accuracy_score, recall_score, f1_score
import torch.optim.lr_scheduler as lr_scheduler

from torch_dataset.RedditDataset import RedditDataset
from torch_dataset.SpamDataset import SpamDataset
from torch_dataset.FinancialDataset import FinancialDataset
from torch_dataset.AmazonDataset import AmazonDataset
from torch_dataset.YoutubeDataset import YoutubeDataset
from torch_dataset.TwitterDataset import TwitterDataset


def train(model, device, dataloader, optimizer, criterion, is_multi):
    model.train()
    accuracy = 0.0
    recall = 0.0
    f1 = 0.0
    total_loss = 0.0
    for batch in dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()

        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        preds = torch.argmax(logits, dim=1)
        loss = criterion(logits, labels)
        total_loss += loss.item()
        # 将张量转换为NumPy数组
        y_true = labels.cpu().numpy()
        y_pred = preds.cpu().numpy()

        loss.backward()
        optimizer.step()

        if is_multi:
            # 计算准确率和召回率
            accuracy += accuracy_score(y_true, y_pred)
            recall += recall_score(y_true, y_pred, average='macro', zero_division=1)
            f1 += f1_score(y_true, y_pred, average='macro', zero_division=1)
        else:
            accuracy += accuracy_score(y_true, y_pred)
            recall += recall_score(y_true, y_pred)
            f1 += f1_score(y_true, y_pred)

    return total_loss / len(dataloader), accuracy / len(dataloader), recall / len(dataloader), f1 / len(dataloader)


def validate(model, device, dataloader, criterion, is_multi):
    model.eval()
    accuracy = 0.0
    recall = 0.0
    f1 = 0.0
    total_loss = 0.0
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1)
            loss = criterion(logits, labels)
            total_loss += loss.item()

            # 将张量转换为NumPy数组
            y_true = labels.cpu().numpy()
            y_pred = preds.cpu().numpy()

            if is_multi:
                # 计算准确率和召回率
                accuracy += accuracy_score(y_true, y_pred)
                recall += recall_score(y_true, y_pred, average='macro', zero_division=1)
                f1 += f1_score(y_true, y_pred, average='macro', zero_division=1)
            else:
                accuracy += accuracy_score(y_true, y_pred)
                recall += recall_score(y_true, y_pred)
                f1 += f1_score(y_true, y_pred)

    return total_loss / len(dataloader), accuracy / len(dataloader), recall / len(dataloader), f1 / len(dataloader)


num_classes = 3
result_file_path = './result/bert/youtube_result_imbalance.csv'
model_path_name = "./dataroot/models/bert-base-uncased"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
tokenizer = BertTokenizer.from_pretrained(model_path_name)
bert_model = BertForSequenceClassification.from_pretrained(model_path_name, num_labels=num_classes)
bert_model.to(device)
max_len = 128
# dataset = RedditDataset(tokenizer=tokenizer, max_length=max_len)
# trainDataset, validDataset = dru.load_dataset(dataset, split_ratio=0.8)
# warn: modify file_path in BalancedDataset class
dataset = YoutubeDataset(tokenizer=tokenizer, max_length=max_len)
trainDataset, validDataset = dru.load_dataset(dataset, split_ratio=0.8)

num_epochs = 40
batch_size = 128
lr = 3e-5
min_lr = 1e-8
T0 = 10
T_min = 1
alpha = 0.95

optimizer = optim.AdamW(bert_model.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()
scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=T0, T_mult=1, eta_min=min_lr, last_epoch=-1)

train_dataloader = DataLoader(trainDataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(validDataset, batch_size=batch_size, shuffle=False)

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
        train(bert_model, device, train_dataloader, optimizer, criterion, multi_classes)
    val_loss, val_acc, val_recall, val_f1 = validate(bert_model, device, val_dataloader, criterion, multi_classes)
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
