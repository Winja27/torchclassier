# 导入库
import numpy as np
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# 准备数据
y_true = np.array([1, 1, 1, 0, 0, 0]) # 真实标签
y_score = np.array([0.9, 0.8, 0.7, 0.6, 0.4, 0.3]) # 预测概率值

fpr, tpr, thresholds = roc_curve(y_true, y_score)
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], linestyle='--') # 对角线
plt.xlim([0.0, 1.05])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.show()
