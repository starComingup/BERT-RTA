import torch
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset
import dataset_utils as dru


def label_mapping(categories):
    # create LabelEncoder 对象
    label_encoder = LabelEncoder()
    # use LabelEncoder to encoder the categories
    encoded_labels = label_encoder.fit_transform(categories)
    return encoded_labels


class NewsDataset(Dataset):
    def __init__(self, tokenizer, max_length):
        content = dru.load_file("dataset/news_category/News_Category_Dataset_v3.json", has_header=False)
        texts = content['short_description']
        categories = content['category']
        labels = label_mapping(categories)
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
