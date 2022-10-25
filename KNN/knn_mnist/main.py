import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class Knn:
    def __init__(self):
        pass

    def fit(self, X_train, y_train):
        self.Xtr = X_train
        self.ytr = y_train

    def predict(self, k, X_test):
        num_test = X_test.shape[0]
        label_list = []
        # 使用欧拉公式作为距离测量

        for i in range(num_test):
            distances = np.sqrt(np.sum(((self.Xtr - np.tile(X_test[i],
                                                            (self.Xtr.shape[0], 1)))) ** 2, axis=1))
            nearest_k = np.argsort(distances)
            topK = nearest_k[:k]
            class_count = {}
            for i in topK:
                class_count[self.ytr[i]] = class_count.get(self.ytr[i], 0) + 1
            sorted_class_count = sorted(class_count.items(), key=lambda elem: elem[1], reverse=True)
            label_list.append(sorted_class_count[0][0])
        return np.array(label_list)


plt.rcParams["font.sans-serif"] = ["SimHei"]  # 设置字体
plt.rcParams["axes.unicode_minus"] = False  # 该语句解决图像中的“-”负号的乱码问题

batch_size = 100
path = './'
train_datasets = datasets.MNIST(root=path, train=True, download=True)
test_datasets = datasets.MNIST(root=path, train=False, download=True)

# 加载数据
train_loader = DataLoader(train_datasets, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_datasets, batch_size=batch_size, shuffle=True)

# 对训练数据处理
x_train = train_loader.dataset.data.numpy()
x_train = np.reshape(x_train, (x_train.shape[0], -1))
x_train = x_train.astype(np.float)
y_train = train_loader.dataset.targets.numpy()

# 取后1000个测试数据
test_num = 200
x_test = test_loader.dataset.data[-1 * test_num - 1:-1].numpy()
# mean_image = getXmean(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], -1))
x_test = x_test.astype(np.float)
y_test = test_loader.dataset.targets[-1 * test_num - 1:-1].numpy()

acc_k3 = []
acc_k5 = []

# 利用KNN计算识别率
for train_num in range(1000, 8000, 1000):
    print("When train_num is {}".format(train_num))
    for k in [3, 5]:  # 不同K值计算识别率
        x_train_real = x_train[:train_num]
        y_train_real = y_train[:train_num]

        classifier = Knn()
        classifier.fit(x_train_real, y_train_real)
        y_pred = classifier.predict(k, x_test)
        num_correct = np.sum(y_pred == y_test)
        accuracy = float(num_correct) / test_num
        if k == 3:
            acc_k3.append(accuracy)
        else:
            acc_k5.append(accuracy)

        print('Got %d / %d correct when k= %d => accuracy: %f' % (num_correct, test_num, k, accuracy))

x = [i for i in range(1000, 8000, 1000)]
plt.figure()
plt.xlabel("训练集数量")
plt.ylabel("测试集准确率")
plt.plot(x, acc_k3, label="k=3")
plt.plot(x, acc_k5, label="k=5")
plt.legend()
plt.grid(linestyle="--", alpha=0.5)
plt.title("KNN在MNIST数据集上的表现")
# plt.savefig(fname="figure.svg", format="svg")
plt.show()
