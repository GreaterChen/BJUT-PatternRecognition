import numpy as np






class Knn:
    def __init__(self):
        pass

    def fit(self, X_train, y_train):
        self.Xtr = X_train
        self.ytr = y_train

    def predict(self, k,X_test):
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

