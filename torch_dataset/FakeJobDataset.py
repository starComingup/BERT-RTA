import pandas as pd
import torch
from torch.utils.data import Dataset
import dataset_utils as dru


class FakeJobDataset(Dataset):
    def __init__(self, tokenizer, max_length):
        content = dru.load_file("dataset/fake job posting/fake_job_postings.csv",
                                has_header=True)
        texts = content['description']
        labels = content['Sentiment']
        labels = pd.to_numeric(labels, errors="coerce")
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

        return input_ids, attention_mask, label

