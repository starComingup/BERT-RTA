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

class TwitterNDataset(Dataset):
    def __init__(self, tokenizer, max_length):
        self.class_name=['fun','hate','love','relief','surprise']
        content = dru.load_file("dataset/tweeter n_emotion/tweet_emotions.csv",
                                has_header=True)
        content = content.dropna(axis=0, how="any")  # 剔除标签为空白的样本
        content.reset_index(drop=True, inplace=True)  # 重置索引

        content_filter = content[(content['sentiment'] != 'worry')
                                 & (content['sentiment'] != 'neutral') & (content['sentiment'] != 'empty')
                                 & (content['sentiment'] != 'anger') & (content['sentiment'] != 'boredom')
                                 & (content['sentiment'] != 'enthusiasm') & (content['sentiment'] != 'sadness')
                                 & (content['sentiment'] != 'happiness')].copy()
        content_filter.reset_index(drop=True, inplace=True)
        texts = content_filter['content']
        categories = content_filter['sentiment']
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

    def get_data_label_map(self):
        label_map = {
            0: 'fun',
            1: 'hate',
            2: 'love',
            3: 'relief',
            4: 'surprise'
        }
        return label_map