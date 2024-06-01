import warnings
from gensim.models import Word2Vec
import numpy as np
import os
import random
import jieba


warnings.filterwarnings("ignore")

# 去停用词
def drop_stopwords(context, stop_words_list):
    return [word for word in context if word not in stop_words_list and word != ' ']

# 返回停用词
def text_stop(file_path):
    with open(file_path + 'cn_stopwords.txt', encoding='utf-8') as stop:
        stop_words = stop.read().split("\n")
    stop_words_list = list(stop_words)
    stop_words_list.append("\u3000")
    return stop_words_list

# 文本处理
def process_text_files(file_path, text_names, stop_words_list):
    text_para_jieba = {}
    for file_name in text_names:
        print(f"Processing {file_name}")
        with open(file_path + "/" + file_name + ".txt", "r", encoding='gb18030') as file:
            all_text = file.read()
            for ad in ['本书来自www.cr173.com免费txt小说下载站', '更多更新免费电子书请关注www.cr173.com', '\u3000']:
                all_text = all_text.replace(ad, '')
            paragraphs = all_text.split("\n")
            text_jieba = []
            for para in paragraphs:
                if para.strip() == '':
                    continue
                processed_para = drop_stopwords(jieba.lcut(para), stop_words_list)
                if processed_para:
                    text_jieba.append(processed_para)
            text_para_jieba[file_name] = text_jieba
    return text_para_jieba

if __name__ == '__main__':
    # 读文件
    file_path = r"jyxstxtqj_downcc.com/"
    text_names = ['倚天屠龙记', '天龙八部', '射雕英雄传', '神雕侠侣', '笑傲江湖']

    # 获取停用词列表
    stop_words_list = text_stop(file_path)

    # 处理文本文件
    text_para_jieba = process_text_files(file_path, text_names, stop_words_list)

    # 保存处理后的文本
    output_dir = "output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for name, content in text_para_jieba.items():
        with open(f"{output_dir}/{name}.txt", "w", encoding='utf-8') as f:
            for para in content:
                f.write(" ".join(para) + "\n")

    # 训练Word2Vec模型并输出相似词
    test_name = ['张无忌', '段誉', '郭靖', '杨过', '令狐冲']

    for name in text_names:
        model = Word2Vec(sentences=text_para_jieba[name], vector_size=200, window=5, min_count=10, sg=0, epochs=200)
        print(f"\nWord2Vec model for {name}:")

        for test_word in test_name:
            if test_word in model.wv:
                print(f"\nTop 10 words similar to '{test_word}' in {name}:")
                for result in model.wv.similar_by_word(test_word, topn=10):
                    print(result[0], result[1])