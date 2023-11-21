import torch
from torch.utils.data import Dataset
import dataset_utils as dru


def label_mapping(sentiments):
    label_map = {
        "negative": 2,
        "neutral": 0,
        "positive": 1
    }
    labels = [label_map[sentiment] for sentiment in sentiments]
    return labels


class FinancialDataset(Dataset):
    def __init__(self, tokenizer, max_length):
        content = dru.load_file("dataset/financial sentiment analysis/data.csv",
                                has_header=True)
        texts = content['Sentence']
        sentiments = content['Sentiment']
        labels = label_mapping(sentiments)
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
