import dataset_utils as dru
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer, BertTokenizer, \
    BertForSequenceClassification
from bert_train_val import train, validate
import torch.optim.lr_scheduler as lr_scheduler
import seaborn as sns
import matplotlib.pyplot as plt

from torch_dataset.RedditDataset import RedditDataset
from torch_dataset.SpamDataset import SpamDataset
from torch_dataset.FinancialDataset import FinancialDataset
from torch_dataset.AmazonDataset import AmazonDataset
from torch_dataset.YoutubeDataset import YoutubeDataset
from torch_dataset.TwitterDataset import TwitterDataset
from torch_dataset.TwitterNDataset import TwitterNDataset
from torch_dataset.FakeJobDataset import FakeJobDataset
from torch_dataset.BalancedDataset import BalancedDataset


num_classes = 3
result_file_path = 'result/bert/twitter/selected/result_balanced_t20_5_3.csv'
confusion_pic_save_path = 'result/bert/twitter/selected/balanced_t20_5_3_confusion_matrix.png'
model_path_name = "./dataroot/models/bert-base-uncased"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
tokenizer = BertTokenizer\
    .from_pretrained(model_path_name)
bert_model = BertForSequenceClassification.from_pretrained(model_path_name, num_labels=num_classes)
bert_model.to(device)
max_len = 128
# dataset = RedditDataset(tokenizer=tokenizer, max_length=max_len)
# trainDataset, validDataset = dru.load_dataset(dataset, split_ratio=0.8)
# warn: modify file_path in BalancedDataset class
dataset = BalancedDataset(tokenizer=tokenizer, max_length=max_len)
trainDataset, validDataset = dru.load_dataset(dataset, split_ratio=0.8)

num_epochs = 10
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
    val_loss, val_acc, val_recall, val_f1, conf_mat = validate(bert_model, device, val_dataloader, criterion, multi_classes)

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
