import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import skfuzzy as fuzz

# 购物评价文本
reviews = ["这件商品真是糟糕透了", "质量很好，物有所值", "一般般吧，没有特别的感觉"]

# 对应的购物评分（1-5分）
ratings = np.array([1, 5, 3])

# 模糊化评分
rating_fuzzy = np.zeros((len(ratings), 5))
for i, rating in enumerate(ratings):
    rating_fuzzy[i] = fuzz.interp_membership(np.arange(1, 6), fuzz.trimf(np.arange(1, 6), [rating-1, rating, rating+1]), np.arange(1, 6))

# 定义模糊规则
# 如果评分为3分，评价为消极，则情感为稍微消极
# 如果评分为3分，评价为积极，则情感为稍微积极
rules = [
    np.fmin(rating_fuzzy[:, 2], fuzz.interp_membership(np.arange(0, 1, 0.01), fuzz.trimf(np.arange(0, 1, 0.01), [0, 0.5, 1]), 0.5)),
    np.fmin(rating_fuzzy[:, 2], fuzz.interp_membership(np.arange(0, 1, 0.01), fuzz.trimf(np.arange(0, 1, 0.01), [0.5, 1, 1]), 0.5))
]

# 转换为PyTorch张量
X_train = torch.tensor(rules, dtype=torch.float32)
y_train = torch.tensor([0, 1, 0], dtype=torch.float32)  # 标记为1分和2分的为消极，标记为4分和5分的为积极

# 构建模糊神经网络模型
class FuzzyNN(nn.Module):
    def __init__(self):
        super(FuzzyNN, self).__init__()
        self.fc1 = nn.Linear(2, 10)
        self.fc2 = nn.Linear(10, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        return x

model = FuzzyNN()

# 定义损失函数和优化器
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 训练模型
num_epochs = 10
for epoch in range(num_epochs):
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train.view(-1, 1))
    loss.backward()
    optimizer.step()

# 使用模型进行预测
X_test = torch.tensor([[0.5, 0.5]], dtype=torch.float32)  # 假设评分为3分，评价为一般般吧
y_pred = model(X_test)
print("预测结果：", y_pred.item())
