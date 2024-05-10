if __name__ == '__main__':
    import warnings
    import gensim
    from pprint import pprint
    import gensim.corpora as corpora
    from gensim.models import CoherenceModel
    import numpy as np
    import os
    import random
    import jieba.posseg as psg
    import jieba
    from sklearn.svm import SVC
    import pyLDAvis
    import pyLDAvis.gensim_models
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, classification_report

#返回停用词
    def text_stop(file_path, text_names):
        text_names_id = {}
        for k in range(len(text_names)):
            text_names_id[text_names[k]] = k

        stop = open(file_path + 'cn_stopwords.txt', encoding='utf-8')
        stop_words = stop.read().split("\n")
        stop_words_list = list(stop_words)
        stop_words_list.append("\u3000")
        return stop_words_list

#文本处理
    def process_text_files(file_path, text_names, stop_words_list):
        text_para_jieba = {}
        text_para_char = {}
        for file_name in text_names:
            print(file_name)
            with open(file_path + "/" + file_name + ".txt", "r", encoding='gb18030') as file:
                all_text = file.read()
                all_text = all_text.replace("本书来自www.cr173.com免费txt小说下载站", "")
                all_text = all_text.replace("更多更新免费电子书请关注www.cr173.com", "")
                all_text = all_text.replace("\u3000", "")
                paragraphs = all_text.split("\n")
                text_jieba = []
                text_char = []
                for para in paragraphs:
                    if para == '':
                        continue

                    text_jieba.append(drop_stopwords(jieba.lcut(para), stop_words_list))
                    text_char.append(drop_stopwords([char for char in para], stop_words_list))
                text_para_jieba[file_name] = text_jieba
                text_para_char[file_name] = text_char

        return text_para_jieba, text_para_char

#判断是否为中文
    def is_chinese_words(words):
        for word in words:
            if u'\u4e00' <= word <= u'\u9fa5':
                continue
            else:
                return False
        return True

#分词
    def cut_words(contents):
        cut_contents = jieba.lcut(contents)
        return cut_contents

#去停用词
    def drop_stopwords(context, stop_words_list):
        line_new = []
        for word in context:
            if word in stop_words_list:
                continue
            elif word != ' ':
                line_new.append(word)
        return line_new

#提取数据集和标签
    def extract_paras(text_names, para_num, token_num, text_para_jieba, text_para_char, cut_option):

        text_names_id = {}
        for k in range(len(text_names)):
            text_names_id[text_names[k]] = k

        corpus = []
        src_labels = []
        for file_name in text_names:
            if cut_option == 'word':
                paragraphs = text_para_jieba[file_name]
                curr = []
                for words in paragraphs:
                    curr.extend(words)
                    if len(curr) < token_num:
                        continue
                    else:
                        corpus.append(curr[0:token_num])
                        src_labels.append(file_name)
                        curr = []
            elif cut_option == 'char':
                paragraphs = text_para_char[file_name]
                curr = []
                for words in paragraphs:
                    curr.extend(words)
                    if len(curr) < token_num:
                        continue
                    else:
                        corpus.append(curr[0:token_num])
                        src_labels.append(file_name)
                        curr = []

        dataset = []
        sampled_labels = []
        para_num_per_text = int(para_num / len(text_names)) + 1
        for label in text_names:
            label_paragraphs = [paragraph for paragraph, paragraph_label in zip(corpus, src_labels) if
                                paragraph_label == label]
            if len(label_paragraphs) < para_num_per_text:
                label_paragraphs = label_paragraphs * int(para_num_per_text / len(label_paragraphs) + 1)
            sampled_index_list = np.random.choice(len(label_paragraphs), para_num_per_text, replace=False)

            sampled_paragraphs = []
            for index in sampled_index_list:
                sampled_paragraphs.append(label_paragraphs[index])
            dataset.extend(sampled_paragraphs)
            sampled_labels.extend([text_names_id[label]] * para_num_per_text)

        dataset = dataset[0:para_num]
        sampled_labels = sampled_labels[0:para_num]

        return dataset, sampled_labels


    def main(text_names, token_num, topic_num, text_para_jieba, text_para_char):
        para_num = 1000
# 计算word的准确率
        dataset, labels = extract_paras(text_names, para_num, token_num, text_para_jieba, text_para_char, 'word')
        id2word = corpora.Dictionary(dataset)
        dataset_train, dataset_test, labels_train, labels_test = train_test_split(dataset, labels, test_size=0.1,
                                                                                  random_state=42)

        train_corpus = [id2word.doc2bow(text) for text in dataset_train]
        test_corpus = [id2word.doc2bow(text) for text in dataset_test]
#LDA建模
        lda_model_word = gensim.models.ldamodel.LdaModel(corpus=train_corpus, id2word=id2word, num_topics=topic_num,
                                                    random_state=100, update_every=1, chunksize=1000, passes=10,
                                                    alpha='auto', per_word_topics=True, dtype=np.float64)

        train_cla = []
        test_cla = []
        for i, item in enumerate(test_corpus):
            tmp = lda_model_word.get_document_topics(item)
            init = np.zeros(topic_num)
            for index, v in tmp:
                init[index] = v
            test_cla.append(init)

        for i, item in enumerate(train_corpus):
            tmp = lda_model_word.get_document_topics(item)
            init = np.zeros(topic_num)
            for index, v in tmp:
                init[index] = v
            train_cla.append(init)

        print("word")
        accuracy_word = np.mean(
            cross_val_score(RandomForestClassifier(), train_cla + test_cla, labels_train + labels_test, cv=10))
        print(f'Accuracy: {accuracy_word:.2f}')
# 计算char的准确率
        dataset, labels = extract_paras(text_names, para_num, token_num, text_para_jieba, text_para_char, 'char')
        id2word = corpora.Dictionary(dataset)
        dataset_train, dataset_test, labels_train, labels_test = train_test_split(dataset, labels, test_size=0.1,
                                                                                  random_state=42)
        train_corpus = [id2word.doc2bow(text) for text in dataset_train]
        test_corpus = [id2word.doc2bow(text) for text in dataset_test]

        lda_model_char = gensim.models.ldamodel.LdaModel(corpus=train_corpus, id2word=id2word, num_topics=topic_num,
                                                    random_state=100, update_every=1, chunksize=1000, passes=10,
                                                    alpha='auto', per_word_topics=True, dtype=np.float64)

        train_cla = []
        test_cla = []
        for i, item in enumerate(test_corpus):
            tmp = lda_model_char.get_document_topics(item)
            init = np.zeros(topic_num)
            for index, v in tmp:
                init[index] = v
            test_cla.append(init)

        for i, item in enumerate(train_corpus):
            tmp = lda_model_char.get_document_topics(item)
            init = np.zeros(topic_num)
            for index, v in tmp:
                init[index] = v
            train_cla.append(init)

        print("char")
        accuracy_char = np.mean(
            cross_val_score(RandomForestClassifier(), train_cla + test_cla, labels_train + labels_test, cv=10))
        print(f'Accuracy: {accuracy_char:.2f}')

#读文件
    file_path = r"jyxstxtqj_downcc.com/"
    text_names = ['碧血剑', '飞狐外传', '连城诀', '鹿鼎记', '三十三剑客图', '射雕英雄传', '神雕侠侣',
                  '书剑恩仇录', '天龙八部', '侠客行', '笑傲江湖', '雪山飞狐', '倚天屠龙记']

    stop_words_list = text_stop(file_path, text_names)
    text_para_jieba, text_para_char = process_text_files(file_path, text_names, stop_words_list)

#设置token和topic
    token_list = [20, 100, 500, 1000]
    topic_list = [5, 20, 100, 500]
    for token_n in token_list:
        for topic_n in topic_list:
            print('token=', token_n, " ",'topic=',  topic_n)
            main(text_names, token_n, topic_n, text_para_jieba, text_para_char)