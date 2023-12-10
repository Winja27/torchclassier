import torch
import data
import time
from sklearn.metrics import recall_score
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
start = time.perf_counter()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



# 定义一个CNNModel类
class CNNModel(torch.nn.Module):
    # 定义__init__方法
    def __init__(self):
        # 调用父类的__init__方法
        super(CNNModel, self).__init__()
        # 定义第一个卷积层，输入通道为3（RGB），输出通道为6，
        # 卷积核大小为5*5
        self.conv1 = torch.nn.Conv2d(3, 6, (5, 5))
        # 定义第一个池化层，使用最大池化，池化核大小为2*2，步长为2
        self.pool1 = torch.nn.MaxPool2d((2, 2), 2)

        self.conv2 = torch.nn.Conv2d(6, 16, (2, 2))

        self.pool2 = torch.nn.MaxPool2d(2, 2)
        # 定义第一个全连接层，输入特征为16*30*30
        # （根据卷积和池化后的图像大小计算），输出特征为400
        self.fc1 = torch.nn.Linear(16 * 30 * 30, 400)
        # 定义第二个全连接层，输入特征为400，输出特征为200
        self.fc2 = torch.nn.Linear(400, 200)
        # 定义第三个全连接层，输入特征为200，输出特征为2（分类数）
        self.fc3 = torch.nn.Linear(200, 2)

    # 定义forward方法
    def forward(self, x):
        # 将输入数据通过第一个卷积层和激活函数得到特征图
        x = torch.nn.functional.relu(self.conv1(x))
        # 将特征图通过第一个池化层进行下采样
        x = self.pool1(x)
        # 将下采样后的特征图通过第二个卷积层和激活函数得到新的特征图
        x = torch.nn.functional.relu(self.conv2(x))
        # 将新的特征图通过第二个池化层进行下采样
        x = self.pool2(x)
        # 将下采样后的特征图展平成一维向量
        x = x.view(-1, 16 * 30 * 30)
        # 将一维向量通过第一个全连接层和激活函数得到新的一维向量
        x = torch.nn.functional.relu(self.fc1(x))
        # 将新的一维向量通过第二个全连接层和激活函数得到新的一维向量
        x = torch.nn.functional.relu(self.fc2(x))
        # 将新的一维向量通过第三个全连接层得到最终的输出向量
        x = self.fc3(x)
        return x


# 实例化CNNModel类
net = CNNModel()
net.to(device)
# 定义一个损失函数
criterion = torch.nn.CrossEntropyLoss()
# 定义一个优化器
optimizer = torch.optim.Adam(net.parameters(), lr=0.02)


# 定义一个训练函数
def train(trainloader, epochs):
    # 初始化一个空列表，用来存储每个批次的损失和准确率
    train_losses = []
    train_acc = []
    # 循环遍历指定次数的训练集
    for epoch in range(epochs):
        # 初始化一个变量，用来存储当前批次的总损失和总准确数
        running_loss = 0.0
        running_correct = 0.0
        # 循环遍历当前批次中的每个图片和标签
        for i, data in enumerate(trainloader, 0):
            # 获取图片和标签，并转换为torch变量，并放到合适的设备上（CPU或GPU）
            inputs = data[0].to(device)
            labels = data[1].to(device)
            # 将优化器中的梯度清零
            optimizer.zero_grad()
            # 将图片输入模型，得到输出
            outputs = net(inputs)
            # 使用损失函数计算输出和标签之间的损失
            loss = criterion(outputs, labels)
            # 使用backward方法计算损失对每个参数的梯度
            loss.backward()
            # 使用优化器中的step方法更新参数
            optimizer.step()
            # 计算输出和标签之间的准确率，并将其加入当前批次的总准确数中
            _, predicted = torch.max(outputs.data, 1)
            running_correct += (predicted == labels).sum().item()
            # 将当前批次的损失加入当前批次的总损失中
            running_loss += loss.item()

            fpr, tpr, thresholds = roc_curve(labels.cpu(), predicted.cpu())
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
            plt.plot([0, 1], [0, 1], linestyle='--')  # 对角线
            plt.xlim([0.0, 1.05])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curve')
            plt.legend(loc="lower right")
            plt.show()

            cm = confusion_matrix(labels.cpu(), predicted.cpu())
            print("混淆矩阵为：")
            print(cm)
            recall = recall_score(labels.cpu(), predicted.cpu())
            print("召回率为：", recall)
        # 计算整个训练集上的平均损失和准确率，并打印出来，并将其加入列表中
        train_loss = running_loss / len(trainloader)
        train_accuracy = running_correct / len(trainloader.dataset)
        print('[%d] 平均损失为: %.3f, 训练准确率为: %.3f' % (epoch + 1, train_loss, train_accuracy))
        train_losses.append(train_loss)
        train_acc.append(train_accuracy)
    return train_losses, train_acc


data = data.dataset()
trainloss, trainacc = train(data, 35)
end = time.perf_counter()
print("运行耗时", (end - start)*1000)
