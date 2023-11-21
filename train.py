import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertTokenizer, BertModel

# 加载预训练的BERT tokenizer和模型
tokenizer = BertTokenizer.from_pretrained('./dataroot/models/bert-base-uncased')
bert_model = BertModel.from_pretrained('./dataroot/models/bert-base-uncased')

text = "i love Beijing so much!"
# 将文本数据转化为BERT的词嵌入
tokenized_data = [tokenizer.encode(text, max_length=50, truncation=True, padding='max_length')]

# 使用BERT模型获取词嵌入
bert_outputs = [bert_model(torch.tensor(tokenized_data))[0]]

print(bert_outputs)