import json
import string
import re

import matplotlib.pyplot as plt
'''
1读取文件，并把内容分别写到两个list里
（一个list对应问题集，另一个list对应答案集）
'''
def read_corpus():
    """
    读取给定的语料库，并把问题列表和答案列表分别写入到 qlist, alist 里面。 在此过程中，不用对字符换做任何的处理（这部分需要在 Part 2.3里处理）
    qlist = ["问题1"， “问题2”， “问题3” ....]
    alist = ["答案1", "答案2", "答案3" ....]
    务必要让每一个问题和答案对应起来（下标位置一致）

    """
    path = "F:/Jupyter/data/train-v2.0.json"
    with open(path,'r',encoding='utf8') as f:
        all_data = json.loads(f.read())
    data = all_data['data']

    qlist = []
    alist = []
    '''
    数据格式如下：
    {'title': 'Beyoncé', 
    'paragraphs': [{'qas': [{'question': 'When did Beyonce start becoming popular?', 
                    'id': '56be85543aeaaa14008c9063',
                    'answers': [{'text': 'in the late 1990s', 'answer_start': 269}], 
                    'is_impossible': False}]}
                    
    用来让程序测试这个condition，如果condition为false，那么raise一个AssertionError出来。逻辑上等同于：
    if not condition:
    raise AssertionError()
    '''
    for dic in data:
        paragraphs = dic["paragraphs"]
        for para in paragraphs:
            qas = para["qas"]
            for qa in qas:
                if qa["answers"] != []:

                    answer = qa["answers"][0]["text"]
                    alist.append(answer)
                    question = qa["question"]
                    qlist.append(question)
    assert len(qlist) == len(alist)
    return qlist,alist

'''
2理解数据（可视化分析/统计信息）
对数据的理解是任何AI工作的第一步，需要充分对手上的数据有个更直观的理解。
TODO: 统计一下在qlist 总共出现了多少个单词？ 总共出现了多少个不同的单词？
      这里需要做简单的分词，对于英文我们根据空格来分词即可，其他过滤暂不考虑（只需分词）
'''
'''
  
      这里要盘一下自己不会的  .format 函数  
      一种格式化字符串的函数 str.format()，它增强了字符串格式化的功能。
      基本语法是通过 {} 和 : 来代替以前的 % 。
      ｛｝ 和 ｛0｝ 都是代表着第一个位置，意思是将第一个位置的元素替换过来
      比如：
      print('[{}]'.format(re.escape(string.punctuation)))
      print('[{}]'.format("hello","world"))
      print('[{1}]'.format("hello","world"))
      print('({})'.format("hello","world"))
      
      输出：所以说在｛｝ 外面加上[],()只是加上了一层壳子
      
      print([\!\"\#\$\%\&\'\(\)\*\+\,\-\.\/\:\;\<\=\>\?\@\[\\\]\^_\`\{\|\}\~])
      [hello]
      [world]
      (hello)
      string.punctuation 方法的意思是输出所有的字符串
      re.escape 是将输入序列所有可能是正则表达式符号的符号转义成符号
      re.compile  从compile()函数的定义中，可以看出返回的是一个匹配对象，它单独使用就没有任何意义，需要和findall(), search(), match(）搭配使用。 
      compile()与findall()一起使用，返回一个列表。
      
       counts[word] = counts.get(word,0)+1 
       是对进行计数word出现的频率进行统计，当word不在words时，返
       回值是0，当word在words中时，返回+1，以此进行累计计数。
     

'''

def segmentWords(lst):
    total = 0
    word_dict = {}
    for line in lst:
        # 下面两句是将句子中的所有符号都转换成空
        pattern = re.compile('[{}]'.format(re.escape(string.punctuation)))
        senquence = pattern.sub("",line)
        # 分词
        words = senquence.split()
        for word in words:
            word_dict[word] =word_dict.get(word,0) + 1
            total += 1
    return total,word_dict

if __name__ == "__main__":

    qlist,alist = read_corpus()
    word_total,q_dict = segmentWords(qlist)
    total_diff_word = len(q_dict.keys())
    print(word_total)
    print('总共{}个单词'.format(word_total))
    print('总共{}个不同的单词'.format(total_diff_word))

    '''
    # TODO: 统计一下qlist中每个单词出现的频率，并把这些频率排一下序，
    然后画成plot. 比如总共出现了总共7个不同单词，而且每个单词出现的频率为 4, 5,10,2, 1, 1,1
    #       把频率排序之后就可以得到(从大到小) 10, 5, 4, 2, 1, 1, 1. 然后把这7个数plot即可（从大到小）
    #       需要使用matplotlib里的plot函数。y轴是词频
    '''
    # k = k[1] 按照词数进行排序
    word_sorted = sorted(q_dict.items(),key=lambda k:k[1],reverse=True)
    #print(word_sorted)
    '''
    输出结果类似于：
    [('Chhauni', 11), ('Silkhana', 1), 
    ('Mahendra', 1), ('Narayana', 1), ('Thangkas', 1), ('Srijana', 1), 
    ('Moti', 1), ('Azima', 1), ('Durbarmarg', 1), ('Aarohan', 1), ('Gurukul', 1),
    ('chop', 1), ('suey', 1), ('thwon', 1), ('bhattis', 1),
    ('tongba', 1), ('Ghode', 1), ('Kirants', 1), ('mourner', 1),
    ('Bhrikuti', 1), ('evangelize', 1), ('Mundhum', 1), 
    ('renovating', 1), ('Prithvi', 1), ('Arkefly', 1), ('Yangon', 1), ('Belorussian', 1).....]
    
    key=lambda 元素: 元素[字段索引]
    比如   print(sorted(C, key=lambda x: x[2]))   
    x:x[]字母可以随意修改，排序方式按照中括号[]里面的维度进行排序，[0]按照第一维排序，[2]按照第三维排序
    '''
    word_freq = []
    word_list = []

    for line in word_sorted:
        word_list.append(line[0])
        word_freq.append(line[1])
    #print(word_freq[:100])
    #print(word_list[:100])

    x = range(total_diff_word)
    #'ro'代表小圆圈是红色的
    plt.plot(x,word_freq,'ro')
    plt.ylabel("word frequency")
    plt.show()

    ####看看词频小于50的词的分布
    temp = [n for n in word_freq if n <=50]
    plt.plot(range(len(temp)),temp, color='r',linestyle='-',linewidth=2)
    plt.ylabel("word frequency")
    plt.show()

    # TODO: 在qlist和alist里出现次数最多的TOP 10单词分别是什么？
    a_total, a_dic = segmentWords(alist)
    words_sorted = sorted(a_dic.items(),key=lambda k:k[1],reverse=True)
    word_freq2 = []
    word_list2 = []
    for line in words_sorted:
        word_list2.append(line[0])
        word_freq2.append(line[1])
    print("top 10 word of qlist are: ", word_list[:10])
    print("top 10 word of alist are: ", word_list2[:10])