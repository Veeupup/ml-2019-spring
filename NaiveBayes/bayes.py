# -*- coding: utf-8 -*-
__author__ = 'Vee'

"""
采用朴素贝叶斯 Naive Bayes 
这里由于数据量太大，我们首先针对部分数据进行训练
手动将数据分为训练集和测试集
"""
import numpy as np
import os
import re
import jieba
import itertools
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB


# 词汇表
myVocabList = []


'''获取所有文档单词的集合'''
def createVocabList(dataSet):
    vocabSet = set([])
    for document in dataSet:
        vocabSet = vocabSet | set(document)  # 操作符 | 用于求两个集合的并集
    return list(vocabSet)


def loadDataSet():
    """
    创建数据集合和每个类的标签
    :return:
    """
    docList = []    # 文档列表
    classList = []  # 类别列表
    global dirList   # 类别列表
    i = 0           # 测试，不能全部读完,自己本机的计算能力有限 我们这里首先只对一部分进行测试
    count = 0
    for root, dirs, files in os.walk("./dataset", topdown=False):
        for name in dirs:
            dirList.append(name)
    # print(dirList)
    for root, dirs, files in os.walk("./dataset", topdown=False):
        for name in files:
            try:
                typeName = os.path.join(root).split("/")[2]
                type = dirList.index(typeName)
                wordList = textParse(open(os.path.join(root, name), 'r', encoding='gbk').read())
                docList.append(wordList)
                classList.append(type)
                # print(typeName)
                # print(os.path.join(root, name))
            except:
                # 某些编码问题首先跳过并找出这些有问题的文件
                count+=1
                pass
    global myVocabList
    myVocabList = createVocabList(docList)  # 创建单词集合
    return docList,classList,myVocabList



def textParse(str_doc):
    """
    利用jieba对文本进行分词，返回切词后的list
    正则过滤掉特殊符号、标点、英文、数字等。
    """
    r1 = '[a-zA-Z0-9’!"#$%&\'()*+,-./:;<=>?@，。?★、…【】《》？“”‘’！[\\]^_`{|}~]+'
    str_doc=re.sub(r1, '', str_doc)    # 创建停用词列表
    stwlist = set([line.strip() for line in open('./stopwords.txt', 'r', encoding='utf-8').readlines()])
    sent_list = str_doc.split('\n')    #
    word_2dlist = [rm_tokens(jieba.cut(part), stwlist) for part in sent_list]  # 分词并去停用词
    word_list = list(itertools.chain(*word_2dlist)) # 合并列表
    return word_list


def rm_tokens(words, stwlist):
    """
    去掉一些停用词、数字、特殊符号
    """
    words_list = list(words)
    for i in range(words_list.__len__())[::-1]:
        word = words_list[i]
        if word in stwlist:  # 去除停用词
            words_list.pop(i)
        elif len(word) == 1:  # 去除单个字符
            words_list.pop(i)
        elif word == " ":  # 去除空字符
            words_list.pop(i)
    return words_list


def getAllVocList(dataSet):
    """
    获取所有文档里面的所有词语，除去重复的词语
    :param dataSet: 所有文档的词语列表
    :return: 所有词语的列表
    """
    vocSet = set([])
    for document in dataSet:
        vocSet = vocSet | set(document)
    return list(vocSet)


def bagWords2Vec(vocList, inputSet):
    """
    输入文档词袋模型
    :param vocList: 所有词语的List
    :param inputSet: 输入文档的词语
    :return: 词袋矩阵向量
    """
    # 一开始的时候，所有的位置都是 0
    vec = [0] * len(vocList)
    # 如果这个词语出现在其中，那么对应的位置置为一
    for word in inputSet:
        if word in vocList:
            vec[vocList.index(word)] += 1
    return vec


def trainNB0(trainMatrix, trainCategory):
    """
    朴素贝叶斯模型训练数据优化
    :param trainMatrix:
    :param trainCategory:
    :return:
    """
    numTrainDocs = len(trainMatrix) # 总文件数
    numWords = len(trainMatrix[0]) # 总单词数

    p1Num=p2Num= np.ones(numWords) # 各类为1的矩阵
    p1Denom=p2Denom = 2.0 # 各类特征和
    num1=num2 = 0 # 各类文档数目

    pNumlist=[p1Num,p2Num]
    pDenomlist =[p1Denom,p2Denom]
    Numlist = [num1,num2]

    for i in range(numTrainDocs): # 遍历每篇训练文档
        for j in range(2): # 遍历每个类别
            if trainCategory[i] == j: # 如果在类别下的文档
                pNumlist[j] += trainMatrix[i] # 增加词条计数值
                pDenomlist[j] += sum(trainMatrix[i]) # 增加该类下所有词条计数值
                Numlist[j] +=1 # 该类文档数目加1

    pVect,pi = [],[]
    for index in range(2):
        pVect.append(np.log(pNumlist[index] / pDenomlist[index]))
        pi.append(Numlist[index] / float(numTrainDocs))
    return pVect, pi


'''朴素贝叶斯分类函数,将乘法转换为加法'''
def classifyNB(vec2Classify, pVect,pi):
    # 计算公式  log(P(F1|C))+log(P(F2|C))+....+log(P(Fn|C))+log(P(C))
    bnpi = [] # 文档分类到各类的概率值列表
    for x in range(2):
        bnpi.append(sum(vec2Classify * pVect[x]) + np.log(pi[x]))
    # print([bnp for bnp in bnpi])
    # 分类集合
    reslist = ['Art','Literature']
    # 根据最大概率，选择索引值
    index = [bnpi.index(res) for res in bnpi if res==max(bnpi)]
    return reslist[index[0]] # 返回分类值


'''文档词袋模型，创建矩阵数据'''
def bagOfWords2VecMN(vocabList, inputSet):
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
    return returnVec

# 存储本地数据
def storedata():
    # 3. 计算单词是否出现并创建数据矩阵
    docList,classList,myVocabList = loadDataSet()
    # 计算单词是否出现并创建数据矩阵
    trainMat = []
    for postinDoc in docList:
        trainMat.append(bagOfWords2VecMN(myVocabList, postinDoc))
    res = ""
    for i in range(len(trainMat)):
        res +=' '.join([str(x) for x in trainMat[i]])+' '+str(classList[i])+'\n'
    # print(res[:-1]) # 删除最后一个换行符
    with open('./word-bag.txt','w') as fw:
        fw.write(res[:-1])
    with open('./wordset.txt','w') as fw:
        fw.write(' '.join([str(v) for v in myVocabList]))


def grabdata():
    """
    读取本地的数据
    :return:
    """
    f = open('./word-bag.txt')  # 读取本地文件
    arrayLines = f.readlines()  # 行向量
    tzsize = len(arrayLines[0].split(' ')) - 1  # 列向量，特征个数减1即数据集
    returnMat = np.zeros((len(arrayLines), tzsize))  # 0矩阵数据集
    classLabelVactor = []  # 标签集，特征最后一列

    index = 0
    for line in arrayLines:  # 逐行读取
        listFromLine = line.strip().split(' ')  # 分析数据，空格处理
        # print(listFromLine)
        returnMat[index, :] = listFromLine[0:tzsize]  # 数据集
        classLabelVactor.append(int(listFromLine[-1]))  # 类别标签集
        index += 1
    # print(returnMat,classLabelVactor)
    myVocabList = writewordset()
    return returnMat, classLabelVactor, myVocabList


def writewordset():
    f1 = open('./wordset.txt')
    myVocabList = f1.readline().split(' ')
    for w in myVocabList:
        if w == '':
            myVocabList.remove(w)
    return myVocabList


'''高斯朴素贝叶斯'''
def MyGaussianNB(trainMat='',Classlabels='',testDoc=''):
    # -----sklearn GaussianNB-------
    # 训练数据
    X = np.array(trainMat)
    Y = np.array(Classlabels)
    # 高斯分布
    clf = GaussianNB()
    clf.fit(X, Y)
    # 测试预测结果
    index = clf.predict(testDoc) # 返回索引
    reslist = ['Art','Literature']
    print(reslist[index[0]])


'''多项朴素贝叶斯'''
def MyMultinomialNB(trainMat='',Classlabels='',testDoc=''):
    # -----sklearn MultinomialNB-------
    # 训练数据
    X = np.array(trainMat)
    Y = np.array(Classlabels)
    # 多项朴素贝叶斯
    clf = MultinomialNB()
    clf.fit(X, Y)
    # 测试预测结果
    index = clf.predict(testDoc) # 返回索引
    reslist = ['Art','Literature']
    print(reslist[index[0]])


'''伯努利朴素贝叶斯'''
def MyBernoulliNB(trainMat='',Classlabels='',testDoc=''):
    # -----sklearn BernoulliNB-------
    # 训练数据
    X = np.array(trainMat)
    Y = np.array(Classlabels)
    # 多项朴素贝叶斯
    clf = BernoulliNB()
    clf.fit(X, Y)
    # 测试预测结果
    index = clf.predict(testDoc) # 返回索引
    reslist = ['Art','Literature']
    print(reslist[index[0]])



def testNB():
    trainMat, Classlabels, myVocabList = grabdata()  # 读取训练结果
    for root, dirs, files in os.walk("./test/C4-Literature", topdown=False):
        for name in files:
            typeName = os.path.join(root).split("/")[2]
            print("本文是"+typeName)
            testEntry = textParse(open(os.path.join(root, name), encoding='gbk').read())
            testDoc = np.array(bagOfWords2VecMN(myVocabList, testEntry))  # 测试数据
            b = []
            b.append(testDoc)
            # 测试预测结果
            p1 = MyGaussianNB(trainMat, Classlabels, b)
            p2 = MyMultinomialNB(trainMat, Classlabels, b)
            p3 = MyBernoulliNB(trainMat, Classlabels, b)
            print(p1, p2, p3)
            print("=====")


if __name__ == '__main__':
    """
    进行测试，输出正确率与每次预测成果
    """
    testNB()




