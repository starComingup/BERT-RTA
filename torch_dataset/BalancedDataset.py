import pandas as pd
import torch
from torch.utils.data import Dataset
import dataset_utils as dru
import warnings

class BalancedDataset(Dataset):
    def __init__(self, tokenizer, max_length):
        youtube_file_path = 'dataset/youtube statistic/balanced_data.csv'
        input_file_path = youtube_file_path
        print('input file path is:'+input_file_path)
        content = dru.load_file(input_file_path,
                                has_header=True)
        content = content.dropna(axis=0, how="any")  # 剔除标签为空白的样本
        content.reset_index(drop=True, inplace=True)  # 重置索引
        texts = content['balanced_texts']
        labels = content['balanced_labels']
        labels = pd.to_numeric(labels, errors="coerce").astype(int)
        self.texts = texts
        self.labels = torch.tensor(labels)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        text = str(self.texts[index])
        label = self.labels[index]

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

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': label
        }

