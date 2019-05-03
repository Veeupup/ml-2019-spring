# -*- coding: utf-8 -*-
__author__ = 'Vee'

"""
决策树的构造
采用 ID-3 算法
"""

from math import log
import operator
import pandas as pd


def calcShannonEnt(dataSet):
    """
    计算给定数据集的香农熵
    :param dataset: 数据集
    :return: 香农熵
    """
    numEntries = len(dataSet)   # 总条目数量
    labelCounts = {}            # 不同分类的标签
    for featVec in dataSet:     # 每一个分类出现的次数
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys(): labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key]) / numEntries
        shannonEnt -= prob * log(prob, 2)
    return shannonEnt


def createDataSet():
    """
    从文件中读取数据集
    :return: 数据集，标签
    """
    sheet = pd.read_excel('./data.xlsx', header=None)
    labels = sheet.values[0].tolist()[:-1]
    dataSet = sheet.values[1:].tolist()
    return dataSet, labels


def splitDataSet(dataSet, axis, value):
    """
    按照给定特征划分数据集
    :param dataSet: 数据集
    :param axis: 划分数据集的特征（数组下标）
    :param value: 返回特征的值
    :return:
    """
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis + 1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet


def chooseBestFeatureToSplit(dataSet):
    """
    从标签中选择最好的划分方式
    :param dataSet: 需要采用划分的数据集
    :return: 返回应该划分的标签的下标
    """
    numFeatures = len(dataSet[0]) - 1  # 特征数目
    baseEntropy = calcShannonEnt(dataSet) # 计算当前的香农熵
    bestInfoGain = 0.0;
    bestFeature = -1
    for i in range(numFeatures):        # 计算每个特征香农熵，并且和基础的香农熵做差比较
        featList = [example[i] for example in dataSet]
        uniqueVals = set(featList)
        newEntropy = 0.0
        for value in uniqueVals:        # 计算每个属性的香农熵
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet) / float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)
        infoGain = baseEntropy - newEntropy
        if (infoGain > bestInfoGain):  # 在所有信息增益中求最大值
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature


def majorityCnt(classList):
    """
    如果已经使用了所有的标签，但是仍然不能区分某些数据
    此时使用多数投票的方式来进行分类
    :param classList:
    :return:
    """
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys(): classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


def createTree(dataSet, labels):
    """
     创建决策树
     使用递归算法创建
     直到用尽来所有标签或者每个标签下的所有实例都是相同的分类
     :param dataset: 数据集
     :param labels: 标签
     :return: 分类字典
     """
    classList = [example[-1] for example in dataSet]
    if classList.count(classList[0]) == len(classList): # 每个类别下的标签的实例都是相同的类
        return classList[0]
    if len(dataSet[0]) == 1:                            # 此时用完来所有标签，仍然不能区分某些实例
        return majorityCnt(classList)                   # 遍历所有特征，返回出现次数最多的类别
    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel: {}}
    del (labels[bestFeat])                               # 删除已经使用的标签
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = labels[:]                           # 由于python是引用传递，防止修改原始标签
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels)
    return myTree


if __name__ == '__main__':
    data, labels = createDataSet()
    myTree = createTree(data, labels)
    print(myTree)