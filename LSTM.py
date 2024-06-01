import warnings

warnings.filterwarnings("ignore")

import os
import jieba
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Input
from tensorflow.keras.models import Model
from sklearn.metrics.pairwise import cosine_similarity


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
def process_text_file(file_path, text_name, stop_words_list):
    print(f"Processing {text_name}")
    with open(file_path + "/" + text_name + ".txt", "r", encoding='gb18030') as file:
        all_text = file.read()
        for ad in ['本书来自www.cr173.com免费txt小说下载站', '更多更新免费电子书请关注www.cr173.com',
                   '她', '他', '你', '我', '它', '这', '\u3000']:
            all_text = all_text.replace(ad, '')
        paragraphs = all_text.split("\n")
        text_jieba = []
        for para in paragraphs:
            if para.strip() == '':
                continue
            processed_para = drop_stopwords(jieba.lcut(para), stop_words_list)
            if processed_para:
                text_jieba.append(processed_para)
        return text_jieba


def train_model(text_name, text_data):
    # Tokenize the text
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(text_data)
    sequences = tokenizer.texts_to_sequences(text_data)
    word_index = tokenizer.word_index
    vocab_size = len(word_index) + 1
    max_sequence_length = max(len(seq) for seq in sequences)

    # Pad sequences
    data = pad_sequences(sequences, maxlen=max_sequence_length, padding='post')

    # Define the context window size
    window_size = 2

    # Prepare input and output pairs for training
    inputs = []
    labels = []

    for sequence in sequences:
        for i in range(window_size, len(sequence) - window_size):
            context_words = sequence[i - window_size:i] + sequence[i + 1:i + window_size + 1]
            target_word = sequence[i]
            inputs.append(context_words)
            labels.append(target_word)

    inputs = np.array(inputs)
    labels = np.array(labels)

    # LSTM model
    embedding_dim = 200
    input_length = 2 * window_size

    input_layer = Input(shape=(input_length,))
    embedding_layer = Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=input_length)(
        input_layer)
    lstm_layer = LSTM(128)(embedding_layer)
    output_layer = Dense(vocab_size, activation='softmax')(lstm_layer)

    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

    # Train model
    model.fit(inputs, labels, epochs=20, batch_size=128)

    # Extract word embeddings
    embeddings = model.layers[1].get_weights()[0]

    return tokenizer, embeddings, word_index


if __name__ == '__main__':
    # 读文件
    file_path = r"jyxstxtqj_downcc.com/"
    text_names = ['倚天屠龙记', '天龙八部', '射雕英雄传', '神雕侠侣', '笑傲江湖']

    # 获取停用词列表
    stop_words_list = text_stop(file_path)

    # 保存处理后的文本
    output_dir = "output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Train models and extract embeddings
    models_data = []
    for name in text_names:
        text_data = process_text_file(file_path, name, stop_words_list)
        tokenizer, embeddings, word_index = train_model(name, text_data)
        models_data.append((name, tokenizer, embeddings, word_index))

    # Test names
    test_name_mapping = {
        '倚天屠龙记': '张无忌',
        '天龙八部': '段誉',
        '射雕英雄传': '郭靖',
        '神雕侠侣': '杨过',
        '笑傲江湖': '令狐冲'
    }

    for name, tokenizer, embeddings, word_index in models_data:
        test_word = test_name_mapping[name]
        if test_word in word_index:
            print(f"\nTop 10 words similar to '{test_word}' in '{name}':")
            test_word_index = word_index[test_word]
            test_word_vector = embeddings[test_word_index].reshape(1, -1)
            similarities = cosine_similarity(test_word_vector, embeddings)[0]
            similar_indices = similarities.argsort()[-11:-1][::-1]
            similar_words = [(tokenizer.index_word[idx], similarities[idx]) for idx in similar_indices]
            for word, similarity in similar_words:
                print(word, similarity)