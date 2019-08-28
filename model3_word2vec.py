from functools import reduce
from sklearn.metrics.pairwise import cosine_similarity
import data_procs as dp
import model2_inverted_table as mit

import re
import string
#TODO:基于词向量的文本表示
# 上面所用到的方法论是基于词袋模型（bag-of-words model）。
# 这样的方法论有两个主要的问题：
# 1. 无法计算词语之间的相似度
# 2. 稀疏度很高。
# 考虑采用词向量作为文本的表示。

#求两个set的交集
def intersections(set1,set2):
    '''

    :param set1:
    :param set2:
    :return: intersection() 方法用于返回两个或更多集合中都包含的元素，即交集。
            union()返回并集 这里应该是找如果没有就全输出出来~  这样子就对了
    '''
    return set1.union(set2)

#加载词向量
'''
line: ionia -0.32568 -0.13751 0.82344 -0.22857 0.11411 0.36041........
row: ['ionia', '-0.32568', '-0.13751', '0.82344', '-0.22857', '0.11411', '0.36041', ......]
'''
def loadGlove(path):
    vocab = {}
    embedding = []
    vocab["UNK"] =0
    embedding.append([0]*100)
    file = open(path,'r',encoding='utf8')
    i = 1
    for line in file:
        #print("line:", line)
        row = line.strip().split()
        #print("row:",row)
        vocab[row[0]] = i
        embedding.append(row[1:])
        i+=1
    print("Finish load Glove")
    file.close()
    return vocab,embedding
def word2vec(words,vocab,emb):
    vec = []

    for word in words:
        if word in vocab:
            idx = vocab[word]
            vec.append(emb[idx])
        else:
            idx = 0
            vec.append(emb[idx])#emb[0] = 100个0
    return vec
def top5result_word2vec(input_q,vocabs,emb):
    """
       给定用户输入的问题 input_q, 返回最有可能的TOP 5问题。这里面需要做到以下几点：
       1. 利用倒排表来筛选 candidate
       2. 对于用户的输入 input_q，转换成句子向量
       3. 计算跟每个库里的问题之间的相似度
       4. 找出相似度最高的top5问题的答案
    """

    #问题预处理
    pattern = re.compile('[{}]'.format(re.escape(string.punctuation)))
    sentence = pattern.sub("",input_q)
    sentence = sentence.lower()
    words = sentence.split()
    result = []
    for word in words:
        if word not in dp.stopwords:
            word = "#number" if word.isdigit() else word
            w = dp.stemmer.stem(word)
            result.append(w)
    #输入问题的词向量
    input_q_vec = word2vec(result,vocabs,emb)

    #根据倒排表
    candidates = []
    for word in result:
        if word in mit.inverted_idx:
            ids = mit.inverted_idx[word]
            candidates.append(set(ids))
    candidate_idx = list(reduce(intersections,candidates))

    #计算相似度得分
    # 计算相似度得分
    scores = []
    for i in candidate_idx:
        sentence = dp.new_qlist[i].split()
        vec = word2vec(sentence, vocabs, emb)
        score = cosine_similarity(input_q_vec, vec)[0]
        scores.append((i, score[0]))
    scores_sorted = sorted(scores, key=lambda k: k[1], reverse=True)

    # 根据索引检索top5答案
    answers = []
    i = 0
    for (idx, score) in scores_sorted:
        if i < 5:
            answer = dp.alist[idx]
            answers.append(answer)
        i += 1
    return answers
if __name__ == "__main__":

    path = 'F:\Jupyter\data\glove.6B.100d.txt'

    vocabs,emb=loadGlove(path)
# 读取每一个单词的嵌入。这个是 D*H的矩阵，
# 这里的D是词典库的大小， H是词向量的大小。 这里面我们给定的每个单词的词向量，那句子向量怎么表达？
# 其中，最简单的方式 句子向量 = 词向量的平均（出现在问句里的），
# 如果给定的词没有出现在词典库里，则忽略掉这个词。
    input_ = input("请输入问题：")
    answ = top5result_word2vec(input_,vocabs,emb)
    print(answ)