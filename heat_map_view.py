import torch
from transformers import BertModel, BertTokenizer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns
import matplotlib.pyplot as plt

# 初始化BERT模型和tokenizer
model_name = './dataroot/models/bert-base-uncased'
model = BertModel.from_pretrained(model_name)
tokenizer = BertTokenizer.from_pretrained(model_name)

# 输入句子
sentences = [
    "Go until jurong point, crazy.. Available only in bugis n great world la e buffet... Cine there got amore wat...",
    "Ok lar... Joking wif u oni...",
    "Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005. Text FA to 87121 to receive entry question(std txt rate)T&C's apply 08452810075over18's",
    "U dun say so early hor... U c already then say...",
    "Nah I don't think he goes to usf, he lives around here though",
    "Even my brother is not like to speak with me. They treat me like aids patent.",
    "I'm gonna be home soon and i don't want to talk about this stuff anymore tonight, k? I've cried enough today.",
    "SIX chances to win CASH! From 100 to 20,000 pounds txt> CSH11 and send to 87575. Cost 150p/day, 6days, 16+ TsandCs apply Reply HL 4 info",
    "I've been searching for the right words to thank you for this breather. I promise i wont take your help for granted and will fulfil my promise. You have been wonderful and a blessing at all times.",
    "Eh u remember how 2 spell his name... Yes i did. He v naughty make until i v wet.",
    "Thanks for your subscription to Ringtone UK your mobile will be charged �5/month Please confirm by replying YES or NO. If you reply NO you will not be charged",
    "Did you hear about the new \Divorce Barbie\? It comes with all of Ken's stuff!",
    "As a valued customer, I am pleased to advise you that following recent review of your Mob No. you are awarded with a �1500 Bonus Prize, call 09066364589"
]

# 计算句子的嵌入
def get_sentence_embedding(sentence, model, tokenizer):
    inputs = tokenizer(sentence, return_tensors='pt', truncation=True, padding=True)
    outputs = model(**inputs)
    # 使用最后一个隐藏层的平均值作为句子的嵌入
    last_hidden_state = outputs.last_hidden_state
    sentence_embedding = last_hidden_state.mean(dim=1).squeeze().detach().numpy()
    return sentence_embedding

embeddings = np.array([get_sentence_embedding(sentence, model, tokenizer) for sentence in sentences])

# 计算余弦相似度
similarity_matrix = cosine_similarity(embeddings)

# 画热力图
plt.figure(figsize=(10, 8))
sns.heatmap(similarity_matrix, annot=True, cmap='coolwarm', xticklabels=range(len(sentences)), yticklabels=range(len(sentences)))
plt.title('Cosine Similarity Heatmap')
plt.show()
