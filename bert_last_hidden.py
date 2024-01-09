import torch
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from transformers import BertTokenizer, BertForSequenceClassification

# 批量文本输入
batch_text = [
    "i love this singler forever",
    "i hate the dog very well",
    "i like to do this thing"
]
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
model_path_name = "./dataroot/models/bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_path_name)
model = BertForSequenceClassification.from_pretrained(model_path_name)
model.to(device)


# 使用BERT模型获取嵌入表示
input_ids_batch = [tokenizer.encode(text, add_special_tokens=True, return_tensors='pt') for text in batch_text]


with torch.no_grad():
    outputs = [model(input_ids.to(device), output_hidden_states=True) for input_ids in input_ids_batch]

embeddings = []
# 获取最后一个隐藏层的表示
for output in outputs:
    last_hidden_states = output.hidden_states[-1]
    paragraph_embedding, _ = last_hidden_states.max(dim=1)
    embeddings.append(paragraph_embedding)

# 将嵌入表示转换为NumPy数组
embedding_array = torch.cat(embeddings).cpu().numpy()

# 指定目标维度
target_dimension = 3

# 使用PCA进行降维
pca = PCA(n_components=target_dimension)
pca_result = pca.fit_transform(embedding_array)

print(pca_result)
print(cosine_similarity(pca_result, pca_result))
# pca_result包含了降维后的表示，可以在你的任务中进一步使用
embeddings = torch.cat(embeddings, dim=0)
print(cosine_similarity(embeddings.cpu().numpy(), embeddings.cpu().numpy()))
# print(embeddings.cpu())
#
# simi_batch_size = 2
# num_samples = embeddings.shape[0]
# simi_result = np.zeros((num_samples, num_samples), dtype=np.float32)
#
# for i in range(0, num_samples, simi_batch_size):
#     start = i
#     end = min(i + simi_batch_size, num_samples)
#     batch_embeddings = embeddings[start:end].cpu().numpy()
#     batch_similarity = cosine_similarity(batch_embeddings, embeddings.cpu().numpy())
#     simi_result[start:end] = batch_similarity
# print(simi_result)
#
# embeddings = embeddings.to(device)
# simi_result = torch.nn.functional.cosine_similarity(embeddings.unsqueeze(0), embeddings.unsqueeze(1))
#
# print(simi_result)