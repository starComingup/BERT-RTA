import torch
import torch.nn as nn

class TextCNNModel(nn.Module):
    def __init__(self, input_size, num_filters, filter_sizes, output_size):
        super(TextCNNModel, self).__init__()
        self.embedding = nn.Embedding(input_size, output_size)
        self.convs = nn.ModuleList([nn.Conv2d(1, num_filters, (fs, output_size)) for fs in filter_sizes])
        self.fc = nn.Linear(num_filters * len(filter_sizes), output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.embedding(x)
        x = x.unsqueeze(1)  # 添加通道维度
        conv_outputs = [self.relu(conv(x)).squeeze(3) for conv in self.convs]
        pooled_outputs = [torch.max(conv, dim=2)[0] for conv in conv_outputs]
        cat_output = torch.cat(pooled_outputs, dim=1)
        output = self.fc(cat_output)
        return output