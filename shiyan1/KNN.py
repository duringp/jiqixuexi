import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns  # 如果没有安装，可以用 pip install seaborn
import os

# 读入mnist数据集
m_x = np.loadtxt('../Hands-on-ML-master/Hands-on-ML-master/第3章 k近邻算法/mnist_x', delimiter=' ')
m_y = np.loadtxt('../Hands-on-ML-master/Hands-on-ML-master/第3章 k近邻算法/mnist_y')

# 数据集可视化
data = np.reshape(np.array(m_x[0], dtype=int), [28, 28])
plt.figure()
plt.imshow(data, cmap='gray')

# 将数据集分为训练集和测试集
ratio = 0.8
split = int(len(m_x) * ratio)
# 打乱数据
np.random.seed(0)
idx = np.random.permutation(np.arange(len(m_x)))
m_x = m_x[idx]
m_y = m_y[idx]
x_train, x_test = m_x[:split], m_x[split:]
y_train, y_test = m_y[:split], m_y[split:]

#(1)对数字“0”和“9”进行可视化。
# 找到所有标签为 0 和 9 的索引
idx_0 = np.where(m_y == 0)[0]
idx_9 = np.where(m_y == 9)[0]

# 取出第一个匹配到的样本
# m_x[idx_0[0]] 是第一个数字 0 的 784 维向量
img_0 = np.reshape(m_x[idx_0[0]].astype(int), [28, 28])
img_9 = np.reshape(m_x[idx_9[0]].astype(int), [28, 28])

# 进行可视化展示
plt.figure(figsize=(8, 4))

# 可视化数字0
plt.subplot(1, 2, 1)
plt.title("Label: 0")
plt.imshow(img_0, cmap='gray')

# 可视化数字9
plt.subplot(1, 2, 2)
plt.title("Label: 9")
plt.imshow(img_9, cmap='gray')

plt.show()


# def distance(a, b):
#     return np.sqrt(np.sum(np.square(a - b)))
#(5)用曼哈顿距离代替欧氏距离。
def distance(a, b):
    # 曼哈顿距离：所有维度差值的绝对值之和
    return np.sum(np.abs(a - b))

class KNN:
    def __init__(self, k, label_num):
        self.k = k
        self.label_num = label_num # 类别的数量

    def fit(self, x_train, y_train):
        # 在类中保存训练数据
        self.x_train = x_train
        self.y_train = y_train

    def get_knn_indices(self, x):
        # 获取距离目标样本点最近的K个样本点的标签
        # 计算已知样本的距离
        dis = list(map(lambda a: distance(a, x), self.x_train))
        # 按距离从小到大排序，并得到对应的下标
        knn_indices = np.argsort(dis)
        # 取最近的K个
        knn_indices = knn_indices[:self.k]
        return knn_indices

    # def get_label(self, x):
    #     # 对KNN方法的具体实现，观察K个近邻并使用np.argmax获取其中数量最多的类别
    #     knn_indices = self.get_knn_indices(x)
    #     # 类别计数
    #     label_statistic = np.zeros(shape=[self.label_num])
    #     for index in knn_indices:
    #         label = int(self.y_train[index])
    #         label_statistic[label] += 1
    #     # 返回数量最多的类别
    #     return np.argmax(label_statistic)

    # (4)将投票机制由“一个样本一票”改为“有区别的加权投票（距离越近的权重越大）”。
    def get_label(self, x):
        # 1. 获取最近的 K 个邻居的下标
        knn_indices = self.get_knn_indices(x)
        # 2. 重新计算这 K 个邻居的具体距离，用于计算权重
        # 提示：为了简化，我们在这里直接算这 K 个点的距离
        label_statistic = np.zeros(shape=[self.label_num])

        for index in knn_indices:
            dist = distance(self.x_train[index], x)
            label = int(self.y_train[index])

            # --- 核心修改：加权投票 ---
            # 距离越近，dist 越小，权重 weight 越大
            # 加上 1e-6 是为了防止 dist 为 0 时报错
            weight = 1.0 / (dist + 1e-6)

            # 不再是 +1，而是加上权重
            label_statistic[label] += weight

        # 返回加权后得分最高的类别
        return np.argmax(label_statistic)

    def predict(self, x_test):
        # 预测样本 test_x 的类别
        predicted_test_labels = np.zeros(shape=[len(x_test)], dtype=int)
        for i, x in enumerate(x_test):
            predicted_test_labels[i] = self.get_label(x)
        return predicted_test_labels

#(2)将训练数据集占整个数据集的比例分别调整为90%、70%、50%，观察测试集分类精度的变化。
# 题目要求的三个比例
print("(2)将训练数据集占整个数据集的比例分别调整为90%、70%、50%，观察测试集分类精度的变化。")
for r in [0.9, 0.7, 0.5]:
    # 重新计算切分点并切分
    split = int(len(m_x) * r)
    x_train_sub, x_test_sub = m_x[:split], m_x[split:]
    y_train_sub, y_test_sub = m_y[:split], m_y[split:]

    # 训练与预测 (固定 K=3)
    knn = KNN(k=3, label_num=10)
    knn.fit(x_train_sub, y_train_sub)
    predicted = knn.predict(x_test_sub)

    # 计算精度
    accuracy = np.mean(predicted == y_test_sub)
    print(f'训练比例 {int(r * 100)}%: 精度为 {accuracy * 100:.1f}%')

print("\n(3)将K值分别设定为1至50，测试分类精度，对精度变化趋势进行可视化并分析变化的原因。")
# 固定比例 0.8
split = int(len(m_x) * 0.8)
x_train_k, x_test_k = m_x[:split], m_x[split:]
y_train_k, y_test_k = m_y[:split], m_y[split:]

# (6)定义混淆矩阵函数
def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=range(10), yticklabels=range(10))
    plt.xlabel('Predicted Label (预测值)')
    plt.ylabel('True Label (真实值)')
    plt.title('Confusion Matrix for MNIST KNN')
    plt.show()

ks = range(1, 51)
accuracies = []
for k in ks:
    knn = KNN(k=k, label_num=10)
    knn.fit(x_train_k, y_train_k)
    pred = knn.predict(x_test_k)
    acc = np.mean(pred == y_test_k)
    accuracies.append(acc)
    last_predicted_labels = pred#记录最后一次预测的结果

#(6)画出分类结果的混淆矩阵。
print("正在生成混淆矩阵...")
plot_confusion_matrix(y_test_k, last_predicted_labels)

# 绘制精度随 K 变化的折线图
plt.figure(figsize=(8, 5))
plt.plot(ks, accuracies, 'b-o', markersize=3)
plt.title('Accuracy vs K Value')
plt.xlabel('K')
plt.ylabel('Accuracy')
plt.grid(True)
plt.show()
