import scipy.io as scio  # 读取.mat文件
import numpy as np
from functools import reduce  # 计算连乘用
import operator
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["SimHei"]  # 解决图像乱码


def readData(xfile, yfile, index):
    # 加载文件
    x = scio.loadmat(xfile)
    y = scio.loadmat(yfile)

    x = x.get(xfile[8:-4])
    y = y.get(yfile[8:-4])

    # 二值化
    # 对于原始数据，x范围是[0,255],0是纯黑255是纯白
    # 此处处理为0就是黑，1就是白
    x = np.where(x > 0, 1, 0)

    num = x.shape[0]  # 获取样本量

    x_train = x[0:index]
    y_train = y[0:index]
    x_test = x[num-5000:num]
    y_test = y[num-5000:num]

    return x_train, y_train, x_test, y_test


test_acc = []
train_acc = []
for j in range(20):
    # 读取训练集、测试集
    x_train, y_train, x_test, y_test = readData("dataset/mnist_train.mat", "dataset/mnist_train_labels.mat",
                                                1000 + j * 1000)

    # 计算先验概率
    # 按y把数据分成10组
    px_group = []
    # 构建类似二维数组的结构
    for i in range(10):
        px_group.append(i)
        px_group[i] = []

    # px_group的第一维表示类别[0-9], 第二维表示相应的所有x样例
    for i, y in enumerate(y_train):
        px_group[int(y)].append(x_train[i])

    # py[]即为先验概率

    py = []
    for i in range(10):
        py.append((len(px_group[i]) + 1) / (x_train.shape[0] + 10))  # 拉普拉斯平滑，避免概率0的出现

    py = np.array(py)
    py = py.reshape(-1, 1)

    # 计算类条件概率
    # 一共是10*784个数值，代表每个像素点为白色的概率
    # 对每个像素点来说，黑色的概率为(1-白色概率)，即得到了所有特征的所有可能取值的类条件概率值(2*10*784个)
    pxy = []
    for i in range(10):
        group = np.array(px_group[i])  # i=0,group:(5466,784),即数字为0的所有5466个样本
        group = (np.sum(group, axis=0) + 1) / (len(px_group[i]) + 10)  # group:(784,)
        pxy.append(group)

    pxy = np.array(pxy)


    # 计算 先验概率*类条件概率并比较
    def predict_ber(x):
        result = []
        for i in range(10):
            # 提取出取值为1的特征(取值为0的会置0)
            # 此处temp可认为是一个有784个数的列表
            temp = x * pxy[i]
            resulti = []
            for j in np.where(temp == 0, 1 - pxy[i], temp):  # 若取值为0，即为黑色，此时置概率为(1-pxy[i])对位的概率值；若取值为1，不必改变
                py_pxy = py[i][0] * reduce(operator.mul, j)  # 对784个特征取值(0 or 1)的概率进行连乘再乘上先验概率
                resulti.append(py_pxy)
            result.append(resulti)

        result = np.argmax(result, axis=0)  # 取10类中概率最大的类
        return result


    test_result = predict_ber(x_test)

    acc_test = np.where(test_result == np.squeeze(y_test), 1, 0)
    print("第{0}次测试准确率:{1}%".format(j + 1, np.sum(acc_test) * 100 / len(test_result)))
    test_acc.append(np.sum(acc_test) * 100 / len(test_result))

print(test_acc)
test_num = []
for i in range(20):
    test_num.append(1000 + 1000 * i)

plt.figure()
plt.plot(test_num, test_acc)
plt.xlabel("训练样本数")
plt.ylabel("测试集准确率")
plt.title("朴素贝叶斯在mnist数据集上的表现")
plt.yticks(range(75, 90, 1))
plt.show()
