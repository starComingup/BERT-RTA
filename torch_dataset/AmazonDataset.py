import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import dataset_utils as dru


class AmazonDataset(Dataset):
    def __init__(self, tokenizer, max_length):
        self.class_name=['positive','negative']
        content = dru.load_file("dataset/Amazon Musical Instruments Reviews/Musical_instruments_reviews.csv",
                                has_header=True)
        # clean the dataset while content is blank and label lost.
        content_filter = content[(content['overall'] != 3.0) & (content['overall'] != 4.0)].copy()  # 使用 .copy() 创建副本
        clean_content = content_filter.reset_index(drop=True)
        texts = clean_content['reviewText']
        scores = clean_content['overall']
        labels = pd.to_numeric(scores, errors="coerce")
        labels = labels.replace(2, 1)
        labels = labels.replace(5, 0)
        self.texts = texts
        self.labels = torch.tensor(labels, dtype=torch.int64)
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

    def get_class_counts(self):
        # count every class of label
        class_counts = {}
        for label in self.labels:
            if label.item() in class_counts:
                class_counts[label.item()] += 1
            else:
                class_counts[label.item()] = 1
        return class_counts

    def append_data(self, new_data, new_labels):
        # 向数据集追加新数据和标签
        self.texts.extend(new_data)
        new_label_tensors = torch.tensor(new_labels)
        self.labels = torch.cat((self.labels, new_label_tensors), dim=0)

    def get_text_by_category(self, category):
        category_indices = [i for i, label in enumerate(self.labels) if label == category]
        category_texts = [self.texts[i] for i in category_indices]
        return category_texts

    def get_data_label_map(self):
        label_map = {
            0: "good",
            1: "bad"
        }
        return label_map