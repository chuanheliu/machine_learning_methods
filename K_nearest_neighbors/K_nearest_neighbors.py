# -*- coding: utf-8 -*- 2


import csv
import random
import math
import operator

def load_dataset(filename, split, training_set=[], test_set=[]):
    """
    加载数据集 并随机分配到训练集和测试集，split由于random可以 大致表示区分数据的比例
    :param filename:文件名 
    :param split: 要将数据集拆分两部分：训练集，测试集 拆分参数
    :param training_set: 训练集
    :param test_set: 测试集
    :return: 
    """
    # rb是读写的mode（模式）
    with open(filename, 'rb') as csvfile:
        lines= csv.reader(csvfile)  # 读
        dataset = list(lines)   # 转换成list的数据结构

        for x in range(len(dataset)-1):
            # 四个属性
            for y in range(4):
                dataset[x][y] = float(dataset[x][y])

            # 这里随机量和split比较 来分成两部分数据集 random[0,1)
            if random.random() < split:
                training_set.append(dataset[x])
            else:
                test_set.append(dataset[x])


def euclidean_distance(instance1, instance2, length):
    """
    求两个实例的欧氏距离。
    :param instance1:实例1
    :param instance2:实例2
    :param length:维度
    :return:欧氏距离
    """

    distace = 0

    for x in range(length):
        distace += pow((instance1[x] - instance2[x]),2)

    return math.sqrt(distace)

def get_neighbors(training_set, test_instance, k):
    distances = []
    length = len(test_instance)-1
    # 算所有的距离
    for x in range(len(training_set)):
        dist = euclidean_distance(test_instance, training_set[x], length)
        distances.append((training_set[x], dist))


    # 从小到大排序
    distances.sort(key=operator.itemgetter(1))

    neighbors = []
    for x in range(k):
        # 取数前k个在training_set中距离最小的。邻居：training_set[x]
        neighbors.append(distances[x][0])

    return neighbors


def get_result(neighbors):
    """
    根据邻居，投票，返回结果
    :param neighbors: 邻居的二维list 包含实例还有距离
    :return: 投票结果，0.1.2...
    """

    # 一个字典， 结果map数量
    class_votes = {}
    for x in range(len(neighbors)):

        # kind是结果种类0，1，2...
        kind = neighbors[x][-1]   # -1返回的是list最后一个值
        if kind in class_votes:
            class_votes[kind] += 1
        else:
            class_votes[kind] = 1

    sorted_votes = sorted(class_votes.iteritems(),key=operator.itemgetter(1), reverse=True)

    return sorted_votes[0][0]


def get_accuracy(test_set, predictions):
    """
    根据测试集以及结果来测试正确率
    :param test_set:
    :param predictions: 测试结果
    :return:
    """
    correct = 0
    for x in range(len(test_set)):
        if test_set[x][-1] == predictions[x]:
            correct += 1
    return (correct/float(len(test_set)))*100.0


def main():
    training_set = []
    test_set = []
    split = 0.67

    load_dataset(r'iris.data.txt', split, training_set, test_set)

    # str()一般是将数值转成字符串,epr()是将一个对象转成字符串显示，注意只是显示用
    print "Train set: ", repr(len(training_set))
    print "Test set: ", repr(len(test_set))

    predictions = []
    k = 3
    for x in range(len(test_set)):
        neighbors = get_neighbors(training_set, test_set[x], k)
        result = get_result(neighbors)
        predictions.append(result)

        print "> preditved= " + repr(result), ', acual=', repr(test_set[x][-1])

    accuracy = get_accuracy(test_set, predictions)
    print ">>>> Accuracy= " + repr(accuracy), '%'

main()



















