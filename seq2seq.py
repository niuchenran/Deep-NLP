import jieba
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tensorflow as tf
from tensorflow.keras import optimizers
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow import keras
import random
import re

DATA_PATH = '../jyxstxtqj/'

def get_single_corpus(file_path):
    r1 = u'[a-zA-Z0-9’!"#$%&\'()*+,-./:：;<=>?@★、…【】《》‘’[\\]^_`{|}~「」『』（）]+'
    with open(file_path, 'r', encoding='ANSI') as f:
        corpus = f.read()
        corpus = re.sub(r1, '', corpus)
        corpus = corpus.replace('\n', '')
        corpus = corpus.replace('\u3000', '')
        corpus = corpus.replace('本书来自免费小说下载站更多更新免费电子书请关注', '')
        f.close()
    words = list(jieba.cut(corpus))
    print("Corpus length: {}".format(len(words)))
    return words

def generate_sequences(data, max_len=60, step=3):
    sentences = []
    next_tokens = []

    tokens = list(set(data))
    tokens_indices = {token: tokens.index(token) for token in tokens}
    print('Unique tokens:', len(tokens))

    for i in range(0, len(data) - max_len, step):
        sentences.append(
            list(map(lambda t: tokens_indices[t], data[i: i + max_len])))
        next_tokens.append(tokens_indices[data[i + max_len]])
    print('Number of sequences:', len(sentences))

    return sentences, next_tokens, tokens, tokens_indices

def vectorize_labels(next_tokens, num_tokens):
    print('Vectorization...')
    next_tokens_one_hot = []
    for i in next_tokens:
        y = np.zeros((num_tokens,))
        y[i] = 1
        next_tokens_one_hot.append(y)

    return next_tokens_one_hot

def get_dataset(data, max_len=60, step=3):
    sentences, next_tokens, tokens, tokens_indices = generate_sequences(data, max_len, step)
    next_tokens_one_hot = vectorize_labels(next_tokens, len(tokens))
    return sentences, next_tokens_one_hot, tokens, tokens_indices

callbacks_list = [
    keras.callbacks.ModelCheckpoint(
        filepath='text_gen.keras',
        monitor='loss',
        save_best_only=True,
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor='loss',
        factor=0.5,
        patience=1,
    ),
    keras.callbacks.EarlyStopping(
        monitor='loss',
        patience=3,
    ),
]

class SeqToSeq(nn.Module):
    def __init__(self, len_token, embedding_size):
        super(SeqToSeq, self).__init__()
        self.encode = nn.Embedding(len_token, embedding_size)
        self.lstm = nn.LSTM(embedding_size, embedding_size, 2, batch_first=True)
        self.decode = nn.Sequential(
            nn.Linear(embedding_size, len_token),
            nn.Sigmoid()
        )

    def forward(self, x):
        print(x.shape)
        em = self.encode(x).unsqueeze(dim=1)
        print(em.shape)
        mid, _ = self.lstm(em)
        print(mid[:,0,:].shape)
        res = self.decode(mid[:, 0, :])
        print(res.shape)
        return res

def sample(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

def train(x, y, tokens, tokens_indices, epochs=200):
    x = np.asarray(x)
    y = np.asarray(y)
    dataset = tf.data.Dataset.from_tensor_slices((x, y))
    dataset = dataset.shuffle(buffer_size=4096)
    dataset = dataset.batch(128)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    model = models.Sequential([
        layers.Embedding(len(tokens), 256),
        layers.LSTM(256),
        layers.Dense(len(tokens), activation='softmax')
    ])

    optimizer = optimizers.RMSprop(learning_rate=0.1)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer)

    for e in range(epochs):
        model.fit(dataset, epochs=1, callbacks=callbacks_list)

        text = '锦衫剑士突然发足疾奔，绕着青衣剑士的溜溜的转动，脚下越来越快。青衣剑士凝视敌手长剑剑尖，敌剑一动，便挥剑击落。锦衫剑士忽而左转，忽而右转，身法变幻不定。青衣剑士给他转得微感晕眩，喝道：“你是比剑，还是逃命？”刷刷两剑，直削过去。但锦衫剑士奔转甚急，剑到之时，人已离开，敌剑剑锋总是和他身子差了尺许。'
        print(text, end='')
        if e % 20 == 0:
            for temperature in [0.2, 0.5, 1.0, 1.2]:
                text_cut = list(jieba.cut(text))[:60]
                print('\n temperature: ', temperature)
                print(''.join(text_cut), end='')
                for i in range(100):
                    sampled = np.zeros((1, 60))
                    for idx, token in enumerate(text_cut):
                        if token in tokens_indices:
                            sampled[0, idx] = tokens_indices[token]
                    preds = model.predict(sampled, verbose=0)[0]
                    next_index = sample(preds, temperature=1)
                    next_token = tokens[next_index]
                    print(next_token, end='')
                    text_cut = text_cut[1: 60] + [next_token]
                print('\n')

if __name__ == '__main__':
    file = DATA_PATH + '越女剑.txt'
    d = get_single_corpus(file)
    _x, _y, _tokens, _tokens_indices = get_dataset(d)
    train(_x, _y, _tokens, _tokens_indices)