from transformers import BertTokenizer, BertForSequenceClassification
import torch

# Load pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('./dataroot/models/bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('./dataroot/models/bert-base-uncased', num_labels=3)  # Assuming 3 classes

# Input text
input_text = "Your input text here."

# Tokenize and obtain BERT outputs
inputs = tokenizer(input_text, return_tensors="pt")
outputs = model(**inputs)

# The logits tensor contains the scores for each class
logits = outputs.logits

# Apply softmax to get probabilities
probabilities = torch.nn.functional.softmax(logits, dim=1)

# The probabilities tensor now contains the predicted probabilities for each class
print(probabilities)
print(torch.argmax(probabilities, dim=1))