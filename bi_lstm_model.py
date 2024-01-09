import torch
import torch.nn as nn

class BiLSTM(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, output_dim, dropout):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.embedding = nn.EmbeddingBag(len(embedding_weights), embedding_dim, sparse=False)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, bidirectional=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text):
        embedded = self.dropout(self.embedding(text))
        outputs, (hidden, cell) = self.lstm(embedded)
        hidden = self.dropout(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))
        return self.fc(hidden)