import torch
from torch.utils.data import Dataset
import dataset_utils as dru
from sklearn.model_selection import train_test_split


def label_mapping(sentiments):
    label_map = {
        "negative": 2,
        "neutral": 0,
        "positive": 1
    }
    labels = [label_map[sentiment] for sentiment in sentiments]
    return labels

# 将文本转换为Word Embeddings
def get_embeddings(text):
    with torch.no_grad():
        tokens = [token.text for token in nlp(text)]
        embeddings = [embedding_weights[token.vocab.stoi[token]] for token in tokens]
        return torch.tensor(embeddings)

class FinancialDataset(Dataset):
    def __init__(self, nlp):
        content = dru.load_file("dataset/financial sentiment analysis/data.csv",
                                has_header=True)
        content = content.dropna(axis=0, how="any")
        content.reset_index(drop=True, inplace=True)

        self.texts = content['Sentence']
        sentiments = content['Sentiment']
        self.labels = label_mapping(sentiments)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]