from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# 替换以下行中的y_true和y_pred为您的实际标签和模型预测
y_true = [1,1,1,1,1,0,0,0,0,0]  # 实际标签
y_pred = [0,1,1,0,0,0,1,0,1,0]  # 模型预测

# 计算混淆矩阵
cm = confusion_matrix(y_true, y_pred)

# 使用热图可视化混淆矩阵
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()