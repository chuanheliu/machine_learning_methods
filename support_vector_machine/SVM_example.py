# -*- coding: utf-8 -*- 2

import numpy as np
import pylab as pl  #画图功能
from sklearn import svm

# 几次运行随机值结果相同。
np.random.seed(0)

# (20,2)-[2,2]: 20行2列矩阵，均值2方差2的正态分布，-表明在负方面，+表明在正方面
x = np.r_[np.random.randn(20, 2) - [2, 2], np.random.randn(20, 2) + [2, 2]]

# 给随机的点，对应label 前20个是0， 后20个是1
y = [0] * 20 + [1] * 20

# 建模
classifier = svm.SVC(kernel='liner')
classifier.fit(x, y)


# 由于模型已经建立好，实际上函数已经产生，所以以下步骤只是取出超平面的系数
w = classifier.coef_[o]
# 斜率
a = -w[0] / w[1]
# -5 -4 -3 .... 4 5
xx = np.linspace(-5,5)
# 点斜式，后面是截距
yy = a * xx - classifier.intercept_[0] / w[1]

# 建立两条边界线，边界与中心线斜率是相同的，改变截距就可以
b = classifier.support_vectors_[0]
yy_down = a * xx + (b[1] - a * b[0])
b = classifier.support_vectors_[-1]
yy_up = a * xx + (b[1] - a * b[0])


pl.plot(xx, yy, 'k-')




















