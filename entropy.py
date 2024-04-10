import os
import re
import math
import jieba
import logging
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置字体为中文字体，这里以宋体为例
# 一元词频统计
def  get_unigram_tf(tf_dic, words):
    for i in range(len(words)):
        tf_dic[words[i]] = tf_dic.get(words[i], 0) + 1
# 二元模型词频统计
def  get_bigram_tf(tf_dic, words):
    for i in range(len(words)-1):
        tf_dic[(words[i], words[i+1])] = tf_dic.get((words[i], words[i+1]), 0) +1
# 三元模型词频统计
def  get_trigram_tf(tf_dic, words):
    for i in range(len(words)-2):
        tf_dic[((words[i], words[i+1]), words[i+2])] = tf_dic.get(((words[i], words[i+1]), words[i+2]), 0) + 1

def process_file(file_path):
    def preprocess_text(text):
        # 删除所有的隐藏符号
        cleaned_text = ''.join(char for char in text if char.isprintable())
        # 删除所有的非中文字符
        cleaned_text = re.sub(r'[^\u4e00-\u9fa5]', '', cleaned_text)
        # 删除停用词
        with open(file_path, 'r', encoding='gb18030') as f:
            stopwords = set([line.strip() for line in f])
            cleaned_text = ''.join(char for char in cleaned_text if char not in stopwords)
        # 删除所有的标点符号
        punctuation_pattern = re.compile(r'[\s!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~]+')
        cleaned_text = punctuation_pattern.sub('', cleaned_text)
        return cleaned_text

    corpus = []
    count_all = 0
    print(file_path)

    with open(file_path, 'r', encoding='gb18030') as f:
        for line in f:
            if line != '\n':
                processed_line = preprocess_text(line.strip())
                corpus.append(processed_line)
                count_all += len(processed_line)

    split_words = []
    words_len = 0
    line_count = 0  # 已处理的行数
    unigram_tf = {}
    bigram_tf = {}
    trigram_tf = {}
    mode = 'jieba'

    for line in corpus:
        if mode == 'jieba':
            for x in jieba.cut(line):
                split_words.append(x)
                words_len += 1
        else:
            for x in line:
                split_words.append(x)
                words_len += 1

        get_unigram_tf(unigram_tf, split_words)
        get_bigram_tf(bigram_tf, split_words)
        get_trigram_tf(trigram_tf, split_words)
        split_words = []
        line_count += 1

    print("语料库字数:", count_all)
    print("分词个数:", words_len)
    print("平均词长:", round(count_all / words_len, 5))

    # 部分词频展示
    tf_dic_list = sorted(unigram_tf.items(), key=lambda x: x[1], reverse=True)
    for i in range(0, 5):
        print(tf_dic_list[i][0], tf_dic_list[i][1])
    bigram_tf_list = sorted(bigram_tf.items(), key=lambda x: x[1], reverse=True)
    for i in range(0, 5):
        print(bigram_tf_list[i][0], bigram_tf_list[i][1])
    trigram_tf_list = sorted(trigram_tf.items(), key=lambda x: x[1], reverse=True)
    for i in range(0, 5):
        print(trigram_tf_list[i][0], trigram_tf_list[i][1])

    words_len = sum([dic[1] for dic in unigram_tf.items()])
    bigram_len = sum([dic[1] for dic in bigram_tf.items()])
    trigram_len = sum([dic[1] for dic in trigram_tf.items()])
    print("一元模型个数:", words_len)
    print("二元模型个数:", bigram_len)
    print("三元模型个数:", trigram_len)

    entropy1 = []
    entropy1 = [-(uni_word[1] / words_len) * math.log(uni_word[1] / words_len, 2) for uni_word in unigram_tf.items()]
    print("基于jieba分割的一元模型的中文信息熵为:", round(sum(entropy1), 5), "比特/词")
    print("基于jieba分割的一元模型的中文平均信息熵为:", round(sum(entropy1) / len(entropy1), 5), "比特/词")

    entropy2 = []
    for bigram_word in bigram_tf.items():
        jp_xy = bigram_word[1] / bigram_len  # 计算联合概率p(x,y)
        cp_xy = bigram_word[1] / unigram_tf[bigram_word[0][0]]  # 计算条件概率p(x|y)
        entropy2.append(-jp_xy * math.log(cp_xy, 2))  # 计算二元模型的信息熵
    print("基于jieba分割的二元模型的中文信息熵为:", round(sum(entropy2), 5), "比特/词")
    print("基于jieba分割的二元模型的中文平均信息熵为:", round(sum(entropy2) / len(entropy2), 5), "比特/词")

    entropy3 = []
    for trigram_word in trigram_tf.items():
        jp_xy = trigram_word[1] / trigram_len
        cp_xy = trigram_word[1] / bigram_tf[trigram_word[0][0]]
        entropy3.append(-jp_xy * math.log(cp_xy, 2))
    print("基于jieba分割的三元模型的中文信息熵为:", round(sum(entropy3), 5), "比特/词")
    print("基于jieba分割的三元模型的中文平均信息熵为:", round(sum(entropy3) / len(entropy3), 5), "比特/词")

    return sum(entropy1), sum(entropy2), sum(entropy3)

file_paths = ['碧血剑.txt', '鹿鼎记.txt', '神雕侠侣.txt ', '白马啸西风.txt', '飞狐外传.txt', '连城诀.txt', '三十三剑客图.txt', '射雕英雄传.txt', '书剑恩仇录.txt', '天龙八部.txt']
entropies1 = []
entropies2 = []
entropies3 = []

for file_path in file_paths:
    entropy1, entropy2, entropy3 = process_file(file_path)
    entropies1.append(entropy1)
    entropies2.append(entropy2)
    entropies3.append(entropy3)

# 绘制折线图
plt.plot(file_paths, entropies1, label='Entropy 1')
plt.plot(file_paths, entropies2, label='Entropy 2')
plt.plot(file_paths, entropies3, label='Entropy 3')

plt.xlabel('File Name', fontsize=8)
plt.ylabel('Entropy')
plt.title('Entropy Comparison')


plt.legend()
plt.show()