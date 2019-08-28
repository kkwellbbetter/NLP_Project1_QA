import data_uderstand as du

import re
import matplotlib.pyplot as plt
import string
from  nltk.corpus import stopwords
from  nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer()
stemmer = PorterStemmer()
stopwords = set(stopwords.words('English'))
'''
Part 3 ：文本预处理
次部分需要尝试做文本的处理。在这里我们面对的是英文文本，所以任何对英文适合的技术都可以考虑进来。
# TODO: 对于qlist, alist做文本预处理操作。 可以考虑以下几种操作：
#       1. 停用词过滤 （去网上搜一下 "english stop words list"，会出现很多包含停用词库的网页）   
#       2. 转换成lower_case： 这是一个基本的操作   
#       3. 去掉一些无用的符号： 比如连续的感叹号！！！， 或者一些奇怪的单词。
#       4. 去掉出现频率很低的词：比如出现次数少于10,20....
#       5. 对于数字的处理： 分词完只有有些单词可能就是数字比如44，415，把所有这些数字都看成是一个单词，这个新的单词我们可以定义为 "#number"
#       6. stemming（利用porter stemming): 因为是英文，所以stemming也是可以做的工作
#       7. 其他（如果有的话）
#       请注意，不一定要按照上面的顺序来处理，具体处理的顺序思考一下，然后选择一个合理的顺序


'''
# 预处理：去标点符号，去停用词，stemming,将数字转换为'#number'表示
# qlist = ["问题1"， “问题2”， “问题3” ....]
def preprocessing(lst):
    new_list =[]
    word_dic = {}
    for line in lst:
        pattern = re.compile('[{}]'.format(re.escape(string.punctuation)))
        sentence = pattern.sub("",line)
        # 转换成小写
        sentence = sentence.lower()
        '''
        llist 类似于['In what year did Kathmandu create its initial international relationship?', 'What is KMC an initialism of?']
        '''
        # 将句子中的每个词按照空格分开返回到列表中
        words = sentence.split()
        temp = []
        for word in words:
            if word not in stopwords:
                word = "#number" if word.isdigit() else word
                w = stemmer.stem(word)
                word_dic[w] = word_dic.get(w,0) + 1
                temp.append(w)
                #print(temp)
        new_list.append(temp)
        #print(word_dic)
    return word_dic,new_list

#画出词频统计图
def drawgraph(dic,name):
    freq = [value for value in dic.values()] # or freq = list(dic.value())
    freq.sort(reverse=True)
    temp = [n for n in freq if n <= 50]
    plt.plot(range(len(temp)),temp,'r-')
    plt.ylabel(name)
    plt.show()
#过滤词频小于low而且大于high的词
def filter_w(dic,lst,low,high):
    temp = []
    for k,v in dic.items():
        if v>=low and v<=high:
            temp.append(k)
    new_list = []
    for line in lst:
        words = [w for w in line if w in  temp]
        new_list.append(' '.join(words))
    return new_list

# TODO: 把qlist中的每一个问题字符串转换成tf-idf向量, 转换之后的结果存储在X矩阵里。
#  X的大小是： N* D的矩阵。 这里N是问题的个数（样本个数），
#       D是字典库的大小。
def tf_idf(lst):
    return vectorizer.fit_transform(lst)
#读入文件
#   qlist = ["问题1"， “问题2”， “问题3” ....]
#  alist = ["答案1", "答案2", "答案3" ....]
qlist,alist = du.read_corpus()
#处理过程
#q_dic,q_list 是stemmm之后的数据
#q_list = ["问题1"， “问题2”， “问题3” ....]
q_dic,q_list = preprocessing(qlist)
a_dic,a_list = preprocessing(alist)
new_qlist = filter_w(q_dic,q_list,2,10000)
new_alist = filter_w(a_dic,a_list,2,10000)
X = tf_idf(new_qlist)
if __name__ == "__main__":

    drawgraph(q_dic,"word frequency of qlist")


    drawgraph(a_dic,"word frequency of qlist")



   # print("the length of new qlist is ", len(new_qlist))
    #print("the length of new alist is ", len(new_alist))


    #print(X[:1])
    # TODO: 矩阵X有什么特点？ 计算一下它的稀疏度
    # 稀疏度，即矩阵的密度

    # x_mat = X.toarray()
    # n = len(x_mat)
    # m = len(x_mat[0])
    # t = 0
    # for i in range(n):
    #     for j in range(m):
    #         if x_mat[i][j] != 0:
    #             t+=1
    # sparsity = t/(n*m)
    # print(sparsity)
