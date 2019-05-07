# Naive Bayes 朴素贝叶斯 和 KNN
简单来说，

> ​		朴素贝叶斯 = 贝叶斯公式 + 条件独立假设

​		但是 Naive Bayes 的效果却很好，有些独立假设在各个分类之间的分布都是均匀的所以对于似然的相对大小不产生影响；即便不是如此，也有很大的可能性各个独立假设所产生的消极影响或积极影响互相抵消，最终导致结果受到的影响不大。

​		这里的例子主要是针对中文 NLP 的处理，根据文本内容来分类。



代码主干在 bayes.py 其中有详细的注释，数据在dataset文件夹中。

使用 [jieba](https://github.com/fxsjy/jieba) 分词, 采用 [中文停用词表](https://github.com/goto456/stopwords)。

最终分类模型分别采用了 sklearn 中的三种贝叶斯分类器和 KNN分类器。

