import torch
import torch.nn as nn

class BiLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(BiLSTMModel, self).__init__()
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.bilstm = nn.LSTM(hidden_size, hidden_size, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, _ = self.bilstm(embedded)
        lstm_out = lstm_out.mean(dim=1)  # 池化操作，可以根据任务需求选择其他池化方式
        output = self.fc(self.relu(lstm_out))
        return output