from functools import reduce
from sklearn.metrics.pairwise import cosine_similarity
import data_procs as dp
import re
import string
# 利用倒排表的优化
# 上面的算法，一个最大的缺点是每一个用户问题都需要跟库里的所有的问题都计算相似度。
# 假设我们库里的问题非常多，这将是效率非常低的方法。
# 这里面一个方案是通过倒排表的方式，先从库里面找到跟当前的输入类似的问题描述。
# 然后针对于这些candidates问题再做余弦相似度的计算。这样会节省大量的时间。

def create_inver(lst):
    inverted_idx = {}  # 倒排表
    #遍历question list （记录question 的 index）
    for i in range(len(lst)):
        for word in lst[i].split():
            if word not in inverted_idx.keys():
                inverted_idx[word] = [i] # 记录question 的 index
            else:
                inverted_idx[word].append(i)
    for k in inverted_idx:
        inverted_idx[k] = sorted(inverted_idx[k])
    return inverted_idx
#求两个set的交集
def intersections(set1,set2):
    '''

    :param set1:
    :param set2:
    :return: intersection() 方法用于返回两个或更多集合中都包含的元素，即交集。
            union()返回并集 这里应该是找如果没有就全输出出来~  这样子就对了
    '''
    return set1.union(set2)

def top5results_invidx(input_q,inverted_idx):

    """
        给定用户输入的问题 input_q, 返回最有可能的TOP 5问题。这里面需要做到以下几点：
        1. 利用倒排表来筛选 candidate
        2. 对于用户的输入 input_q 首先做一系列的预处理，然后再转换成tf-idf向量（利用上面的vectorizer)
        3. 计算跟每个库里的问题之间的相似度
        4. 找出相似度最高的top5问题的答案
    """
    #输入的预处理
    pattern = re.compile('[{}]'.format(re.escape(string.punctuation)))
    sentence = pattern.sub("",input_q)
    sentence = sentence.lower()
    words = sentence.split()
    result = []
    for word in words:
        if word not in dp.stopwords:
            word  = "#number" if word.isdigit() else word
            w = dp.stemmer.stem(word)
            result.append(w)

    # print(result)

    #根据倒排表选出问题索引
    candidates = []
    for word in result:
        if word in inverted_idx.keys():
            ids = inverted_idx[word]# ids 里面很多东西 具体看上面解释
            candidates.append(set(ids))

    # print(candidates)
    try:
        r_out = reduce(intersections,candidates)# 候选问题索引 一维
    except :
        print("")
    else:
        candidates_idx = list(r_out)# 候选问题索引 一维

       #print(candidates_idx)# 看看有多少交集，~~

    #如果没有候选词 直接返回错误
    if len(candidates) == 0:
        return print("您输入的问题暂时无法回答呢~")

    '''
    reduce() 函数会对参数序列中元素进行累积。
    函数将一个数据集合（链表，元组等）中的所有数据进行下列操作：
    用传给 reduce 中的函数 function（有两个参数）先对集合中的第 1、2 个元素进行操作，
    得到的结果再与第三个数据用 function 函数运算，最后得到一个结果。
    在这里是求集合的交集，具体看倒排表的原理
    '''
    #将输入转换成向量
    input_seg = ' '.join(result)
    input_vec = dp.vectorizer.transform([input_seg])

    #与每个候选问题计算相似度：这里是最重点的
    res = []

    for i in candidates_idx:
        score = cosine_similarity(input_vec,dp.X[i])[0]
        res.append((i,score))
    res_sorted = sorted(res,key= lambda k:k[1],reverse=True)# 逆序排序

    #print(res_sorted)

    #索引出Top5的答案
    answers = []
    i = 0
    for (idx,score) in res_sorted:
        if i <5:
            answer = dp.alist[idx]
            answers.append(answer)
        i+=1
        #print(i)

    #print(answers)
    return answers
inverted_idx = create_inver(dp.new_qlist)
if __name__ == "__main__":
    while True:
        input_ = input("请输入问题:")
        if input_ != "exit":
            answ = top5results_invidx(input_,inverted_idx)
            print(answ)
        else:
            break



