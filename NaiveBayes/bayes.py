# -*- coding: utf-8 -*-
__author__ = 'Vee'

"""
采用朴素贝叶斯 Naive Bayes 和 KNN 作为分类器
这里由于数据量太大，我们首先针对部分数据进行训练
手动将数据分为训练集和测试集
"""
import numpy as np
import os
import re
import jieba
import itertools
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.neighbors import KNeighborsClassifier


# 词汇表
myVocabList = []
# 类别列表
reslist = ['C3-Art', 'C4-Literature', 'C34-Economy', 'C23-Mine', 'C39-Sports', 'C7-History', 'C19-Computer', 'C29-Transport', 'C15-Energy', 'C38-Politics', 'C11-Space', 'C17-Communication', 'C32-Agriculture', 'C5-Education', 'C37-Military', 'C16-Electronics', 'C36-Medical', 'C6-Philosophy', 'C35-Law', 'C31-Enviornment']


def createVocabList(dataSet):
    """
    获取所有文章中的词语（不重复）
    :return: list
    """
    vocabSet = set([])
    for document in dataSet:
        vocabSet = vocabSet | set(document)  # 操作符 | 用于求两个集合的并集
    return list(vocabSet)


def loadDataSet():
    """
    创建数据集合和每个类的标签
    """
    docList = []    # 文档列表
    classList = []  # 类别列表
    dirList = []   # 类别列表
    count = 0
    for root, dirs, files in os.walk("./train", topdown=False):
        for name in dirs:
            dirList.append(name)
    print(dirList)
    global reslist
    reslist = dirList
    # print(dirList)
    for root, dirs, files in os.walk("./train", topdown=False):
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


def classifyNB(vec2Classify, pVect,pi):
    """
    朴素贝叶斯分类函数,将乘法转换为加法
    """
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


def bagOfWords2VecMN(vocabList, inputSet):
    """
    词袋模型，返回矩阵
    """
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
    return returnVec

# 存储本地数据
def storedata():
    # 3. 计算单词是否出现并创建数据矩阵
    docList,classList,myVocabList = loadDataSet()
    print("读取完毕")
    # 计算单词是否出现并创建数据矩阵
    print("开始计算向量")
    trainMat = []
    i = 0
    for postinDoc in docList:
        print(i)
        i+=1
        trainMat.append(bagOfWords2VecMN(myVocabList, postinDoc))
    print("向量计算完毕")
    res = ""
    for i in range(len(trainMat)):
        res +=' '.join([str(x) for x in trainMat[i]])+' '+str(classList[i])+'\n'
    print("开始写入文件")
    with open('./word-bag.txt','w') as fw:
        fw.write(res[:-1])
    with open('./wordset.txt','w') as fw:
        fw.write(' '.join([str(v) for v in myVocabList]))
    print("文件写入完毕")


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
    global reslist
    list = reslist
    return list[index[0]]


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
    global reslist
    list = reslist
    return list[index[0]]


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
    global reslist
    list = reslist
    return list[index[0]]


def MyKnnClassfier(trainMat='', Classlabels='', testDoc=''):
    # 训练数据
    X = np.array(trainMat)
    Y = np.array(Classlabels)

    knnclf = KNeighborsClassifier()  # default with k=5
    knnclf.fit(X, Y)
    index = knnclf.predict(testDoc)
    global reslist
    list = reslist
    return list[index[0]]


def test():
    print("开始读取数据")
    trainMat, Classlabels, myVocabList = grabdata()  # 读取训练结果
    print("数据读取完毕")
    # 统计三种贝叶斯模型的预测正确的个数
    i1 = 0
    i2 = 0
    i3 = 0
    i4 = 0
    count = 0
    error = 0
    for root, dirs, files in os.walk("./test0", topdown=False):
        for name in files:
            try:
                typeName = os.path.join(root).split("/")[2]
                print("本文是"+typeName)
                print(os.path.join(root, name))
                testEntry = textParse(open(os.path.join(root, name), encoding='gbk').read())
                testDoc = np.array(bagOfWords2VecMN(myVocabList, testEntry))  # 测试数据
                b = []
                b.append(testDoc)
                # 测试预测结果
                p1 = MyGaussianNB(trainMat, Classlabels, b)
                p2 = MyMultinomialNB(trainMat, Classlabels, b)
                p3 = MyBernoulliNB(trainMat, Classlabels, b)
                p4 = MyKnnClassfier(trainMat, Classlabels, b)
                if(typeName == p1):
                    i1 += 1
                if (typeName == p2):
                    i2 += 1
                if (typeName == p3):
                    i3 += 1
                if(typeName == p4):
                    print("yes")
                    i4 += 1
                count+=1
                # print(p1, p2, p3, p4)
                print("=====")
            except Exception:
                print(Exception.args)
                error += 1
    print(i1 / count)
    print(i2 / count)
    print(i3 / count)
    print(error)
    with open('./result.txt','w') as fw:
        fw.write("测试完毕")
        fw.write("三种高斯模型正确率分别为：\n")
        fw.write(str(i1 / count) + "\n")
        fw.write(str(i2 / count) + "\n")
        fw.write(str(i3 / count) + "\n")
        fw.write("KNN的正确率为 \n")
        fw.write(str(i4 / count) + "\n")


if __name__ == '__main__':
    """
    进行测试，输出正确率与每次预测成果
    """
    storedata()
    test()




