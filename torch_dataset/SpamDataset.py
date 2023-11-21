import torch
from torch.utils.data import Dataset
import dataset_utils as dru
import pandas as pd
from sklearn.preprocessing import LabelEncoder


def label_mapping(categories):
    # create LabelEncoder 对象
    label_encoder = LabelEncoder()
    # use LabelEncoder to encoder the categories
    encoded_labels = label_encoder.fit_transform(categories)
    return encoded_labels


class SpamDataset(Dataset):
    def __init__(self, tokenizer, max_length):
        content = dru.load_file("dataset/spam/spam.csv",
                                has_header=True)
        texts = content['v2']
        v1 = content['v1']
        labels = label_mapping(v1)
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
