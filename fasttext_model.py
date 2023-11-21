import torch
import torch.nn as nn


class FastTextModel(nn.Module):
    def __init__(self, input_size, embedding_size, output_size):
        super(FastTextModel, self).__init__()
        self.embedding = nn.Embedding(input_size, embedding_size)
        self.fc = nn.Linear(embedding_size, output_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        embedded = self.embedding(x)
        averaged = torch.mean(embedded, dim=1)  # 平均池化
        output = self.fc(averaged)
        return self.softmax(output)