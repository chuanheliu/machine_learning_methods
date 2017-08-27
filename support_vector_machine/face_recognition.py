# -*- coding: utf-8 -*- 2

from time import time
import logging  #打印程序进展信息
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt  #绘制人脸

from sklearn.cross_validation import train_test_split
from sklearn.datasets import fetch_lfw_people
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import RandomizedPCA
from sklearn.svm import SVC


# 打印程序进展信息
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

# -------------------------------------------------------------------
#   加载预处理数据集，得到想要的格式
# -------------------------------------------------------------------

# sklearn现成的函数来下载数据集（名人照片，以及对应label）
lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)

# n_sample:数据集实例的数量， high和width
n_samples, h, w = lfw_people.images.shape

# 特征向量矩阵
X = lfw_people.data
# shape返回矩阵的0行数和1列数，反-------------------------映特征维度
n_features = X.shape[1]

# 获取class label, 返回字典形式数据。在此例子返回不容人的标记
y = lfw_people.target
# 返回所以类别有谁的名字,来区分数据集有多少人
target_names = lfw_people.target_names
# shape 也是行列，0反映---------------------------------label数量
n_classes = target_names.shape[0]

print 'Total dataset size:'
print 'n_sample: %d' % n_samples
print 'n_features: %d' % n_features
print 'n_classes: %d' % n_classes

# 讲提取出来的数据X和y， 分成两部分训练集和测试集.
# train_test_split是sklearn中的包可直接使用，得到以下四部分，分别两个特征矩阵，两个label向量
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

# -------------------------------------------------------------------
#   特征值维度很大，一下用PCA 方法来降维，可以提高准确性
# -------------------------------------------------------------------

# 组成元素数量 是个参数
n_components = 150

print 'Extracting the top %d eigenfaces from %d faces' % (n_components, X_train.shape[0])

# 当前时间
t0 = time()
# 降维算法，对特征向量训练集。 得到pca模型
pca = RandomizedPCA(n_components=n_components, whiten=True).fit(X_train)

print 'done in %0.3fs' % (time() - t0)

# 人脸的特征值
eigenface = pca.components_.reshape((n_components, h, w))

print 'Projecting the input on the eigenface orthonormal basis'

t0 = time()
# 降维之后的矩阵
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)
print 'done in %0.3fs' % (time() - t0)

# -------------------------------------------------------------------
#   降维之后的特征向量，结合一定分类器分类。
# -------------------------------------------------------------------

print 'Fitting the classifier to the training set'
t0 = time()
# C是对错误的惩罚,1乘以e的三次方。gamma是特征点使用比例
# 多值的尝试：由于不知道哪些参数组合比较好，列出来几个值，来搜索最好组合的归类精确度
param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],
              'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1]}

# GridSearchCV实现组合尝试。（图像因此选rbf核函数，自动权重，二维格子结构看那个最好）
classifier = GridSearchCV(SVC(kernel='rbf', class_weight='auto'), param_grid)
# 建模 边际最大超平面
classifier = classifier.fit(X_train_pca, y_train)


print 'Done in %0.3fs' % (time() - t0)
print 'Best estimator found by grid search'
print (classifier.best_estimator_)


# -------------------------------------------------------------------
#   在测试集上评估模型
# -------------------------------------------------------------------

print "Predicting people's names on the test set"
t0 = time()
# 预测测试集分类
y_pred = classifier.predict(X_train_pca)
print 'Done in %0.3fs' % (time() - t0)

# 调用报告：比较真实标签和测试标签y_test, y_pred，并填入姓名
# print (classification_report(y_test, y_pred, target_names=target_names))

# 矩阵来显示真实和测试结果，对角线表示预测和实际是准确的
# print (confusion_matrix(y_test, y_pred, labels=range(n_classes)))


def plot_gallery(image, titles, h, w, n_row=3, n_col=4):
    """显示框"""
    plt.figre(figsize=(1.8*n_col, 2.4*n_row))
    plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
    for i in range(n_row * n_col):
        plt.subplots(n_row, n_col, i+1)
        plt.imshow(image[i].reshape((h, w)), cmap=plt.cm.gray)
        plt.title(titles[i], size=12)
        plt.xticks(())
        plt.yticks(())


def title(y_pred, y_test, target_names, i):
    """对应的名字"""
    pred_name = target_names[y_pred[i]].rsplit(' ', 1)[-1]
    true_name = target_names[y_test[i]].rsplit(' ', 1)[-1]
    return 'Predicted: %s\ntrue:     %s' %(pred_name, true_name)

prediction_titles = [title(y_pred, y_test, target_names, i)
                     for i in range(y_pred.shape[0])]
plot_gallery(X_test, prediction_titles, h, w)

eigenface_titles = ['eigenface %d' % i for i in range(eigenface.shape[0])]
plot_gallery(eigenface, eigenface_titles, h, w)

plt.show()




