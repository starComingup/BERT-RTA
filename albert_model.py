from transformers import AlbertModel, AlbertTokenizer
import torch
import torch.nn as nn

class AlbertModelWrapper(nn.Module):
    def __init__(self, albert_model, num_labels):
        super(AlbertModelWrapper, self).__init__()
        self.albert = albert_model
        self.fc = nn.Linear(self.albert.config.hidden_size, num_labels)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_ids, attention_mask):
        outputs = self.albert(input_ids, attention_mask=attention_mask)
        pooled_output = outputs['pooler_output']
        output = self.fc(pooled_output)
        return self.softmax(output)