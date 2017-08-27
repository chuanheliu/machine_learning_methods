# -*- coding: utf-8 -*- 2

from sklearn import svm

#
# 通过三个点来确定超平面

# (2,3)在平面上方  （1,1) and (2,0)在平面下方
x = [[2, 0], [1, 1], [2, 3]]
# binary class label 两类问题
y = [0, 0, 1]

# 建立分类器，核函数选择线性
classifier = svm.SVC(kernel='linear')

# 建立分类模型：x是实例的list(特征向量矩阵，这个程序就是三个点)，y：x对应的class label
classifier.fit(x, y)

print '----------------classifier 打印参数-----------------'
print classifier, '\n'

print '-------classifier.support_vectors_ 打印所有点-------'
print classifier.support_vectors_, '\n'

print '-----------classifier.support_ 打印index-----------'
print classifier.support_, '\n'

print '-------classifier.n_support_ 分别有几个支持向量-------'
print classifier.n_support_, '\n'

print '-----------classifier.predict 打印分类结果-----------'
print classifier.predict([1, 9])