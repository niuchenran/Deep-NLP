import jieba
import re
import matplotlib.pyplot as plt
from collections import Counter

plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置字体为中文字体，这里以宋体为例
def preprocess_text(text):
    # 删除所有的隐藏符号
    cleaned_text = ''.join(char for char in text if char.isprintable())
    # 删除所有的非中文字符
    cleaned_text = re.sub(r'[^\u4e00-\u9fa5]', '', cleaned_text)
    # 删除停用词
    with open('停用词.txt', 'r', encoding='gb18030') as f:  # 此处为停用词文件名
        stopwords = set([line.strip() for line in f])
        cleaned_text = ''.join(char for char in cleaned_text if char not in stopwords)
    # 删除所有的标点符号
    punctuation_pattern = re.compile(r'[\s!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~]+')
    cleaned_text = punctuation_pattern.sub('', cleaned_text)
    return cleaned_text

file_names = ['碧血剑.txt', '鹿鼎记.txt', '神雕侠侣.txt', '白马啸西风.txt', '飞狐外传.txt', '连城诀.txt', '三十三剑客图.txt', '射雕英雄传.txt', '书剑恩仇录.txt', '天龙八部.txt']

for file_name in file_names:
    counts = []
    with open(file_name, 'r', encoding='gb18030') as f:
        text = f.read()
        words = jieba.lcut(text)
        counts.extend(words)

    counts_dict = Counter(counts)
    sort_list = sorted(counts_dict.values(), reverse=True)

    plt.plot(sort_list, label=file_name)

plt.title('Zipf-Law', fontsize=18)
plt.xlabel('rank', fontsize=18)
plt.ylabel('frequency', fontsize=18)
plt.yscale('log')  # 设置纵坐标的缩放
plt.xscale('log')  # 设置横坐标的缩放
plt.legend()
plt.show()
