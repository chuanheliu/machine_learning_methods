# -*- coding: utf-8 -*- 2

from sklearn.feature_extraction import DictVectorizer
import csv # python 自带包
from sklearn import preprocessing
from sklearn import tree # decision tree
from sklearn.externals.six import StringIO

# 读取csv数据文件
allElectronicsData = open(r'allElectronicsData.csv','rb')

# csv自带reader功能按行读取数据
reader = csv.reader(allElectronicsData)
headers = reader.next()

featureList = []
labelList = []

for row in reader:
    # 每一行将label加进labelList
    labelList.append(row[len(row) - 1])
    # 这里遍历每一行各个属性，加到字典里
    rowDict = {}
    # 从1开始读 0是编号
    for i in range(1,len(row)-1):
        rowDict[headers[i]] = row[i]

    featureList.append(rowDict)

# print headers
# print labelList
# print featureList


# 实例化DictVectorizer
vec = DictVectorizer()
# dummy variable: 转换成需要的01格式，第一组数据一样的为1 其他为0
dummyX = vec.fit_transform(featureList).toarray()

# print str(dummyX)

# 专门处理label的方法，
lb = preprocessing.LabelBinarizer()
dummyY = lb.fit_transform(labelList)

# print dummyY


# 创建分类器,entropy是信息熵的方法：ID3，不同方法调用不同
# 决策树文档：http://scikit-learn.org/stable/modules/classes.html#module-sklearn.tree
classifler = tree.DecisionTreeClassifier(criterion="entropy")
# 建模
classifler = classifler.fit(dummyX, dummyY)

# print str(classifler)


# 文件读写用with open语句
# 创建一个 .dot文件存储决策树信息
# 文件创建好，可以用graphviz来转换pdf直接阅读 commond命令：dot -Tpdf allElectronicsInformationGainDir.dot -o tree.pdf
with open("allElectronicsInformationGainDir.dot",'w') as f:
    # 还原我们转换成的0和1，返回我们变换之前的属性，用：feature_names=vec.get_feature_names()
    f = tree.export_graphviz(classifler, feature_names=vec.get_feature_names(), out_file=f)


#-------------------------------------------------------------------------------------------------------
#   以上就已经把树的模型建好，下面可以使用来进行预测
#-------------------------------------------------------------------------------------------------------
# 第0行 [0,:]   第0列 [:,0]

# 这样创建一个新的人，此处是通过获得第一个人属性然后修改一些属性来获取
oneRowX = dummyX[0,:]

# youth	high	no	fair	no
newRow=oneRowX

# youth	high	yes	fair	no
newRow[0] = 1
newRow[2] = 0

# print newRow
# print oneRowX

# 预测
preditedY = classifler.predict(newRow)
print preditedY