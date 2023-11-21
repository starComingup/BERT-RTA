import torch
from torch.utils.data import Dataset
import dataset_utils as dru
import pandas as pd


class TwitterDataset(Dataset):
    def __init__(self, tokenizer, max_length):
        content = dru.load_file("dataset/twitter sentiment analysis/Twitter_Data.csv",
                                has_header=True)
        content = content.dropna(axis=0, how="any")  # 剔除标签为空白的样本
        content.reset_index(drop=True, inplace=True)  # 重置索引

        proportion_to_keep = 0.1
        # 对每个类别进行层次抽样以保持原分布
        sampled_data = []
        for category, group in content.groupby('category'):
            category_sample = group.sample(frac=proportion_to_keep, random_state=1)
            sampled_data.append(category_sample)

        content_sampled = pd.concat(sampled_data)
        content_sampled.reset_index(drop=True, inplace=True)
        texts = content_sampled['clean_text']
        categories = content_sampled['category']
        labels = pd.to_numeric(categories, errors="coerce").astype(int)
        # Convert -1 to 2 in labels
        labels = labels.replace(-1, 2)
        labels = labels.to_numpy()

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
        # append new data and labels to dataset
        self.texts.extend(new_data)
        new_label_tensors = torch.tensor(new_labels)
        self.labels = torch.cat((self.labels, new_label_tensors), dim=0)

    def get_text_by_category(self, category):
        category_indices = [i for i, label in enumerate(self.labels) if label == category]
        category_texts = [self.texts[i] for i in category_indices]
        return category_texts
