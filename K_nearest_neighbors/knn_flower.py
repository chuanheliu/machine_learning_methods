# -*- coding: utf-8 -*- 2

from sklearn import neighbors
from sklearn import datasets

# KNN的分类器
knn = neighbors.KNeighborsClassifier()

# 返回数据库的数据集,数据內建在dataset数据库里面
iris = datasets.load_iris()

# 花儿的四维特征值。 还有012来区分三种花
# print iris


# 大部分分类器都有fit()这个功能，是用来建立模型的。
# data是特征值矩阵  target是一维向量代表每一行分类结果
knn.fit(iris.data, iris.target)

# 预测新实例[0.1,0.2,0.3,0.4]
predictedLabel = knn.predict([0.1,0.2,0.3,0.4])

print predictedLabel