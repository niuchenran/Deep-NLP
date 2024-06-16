import jieba
import numpy as np
import tensorflow as tf
import keras
from keras import optimizers
from keras import layers
from keras import models
import re

DATA_PATH = 'C:\\Users\\Chen\\Desktop\\buaa_nlp_project5-main\\jyxstxtqj\\'
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


def get_dataset(data):

    max_len = 60
    step = 3
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

    print('Vectorization...')
    next_tokens_one_hot = []
    for i in next_tokens:
        y = np.zeros((len(tokens),))
        y[i] = 1
        next_tokens_one_hot.append(y)
    return sentences, next_tokens_one_hot, tokens, tokens_indices
callbacks_list = [
    tf.keras.callbacks.ModelCheckpoint(
        filepath='text_gen.keras',
        monitor='loss',
        save_best_only=True,
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='loss',
        factor=0.5,
        patience=1,
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor='loss',
        patience=3,
    ),
]


def sample(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

class TransformerEncoder(layers.Layer):
    def __init__(self, num_heads, key_dim, ff_dim, rate=0.1):
        super(TransformerEncoder, self).__init__()
        self.mha = layers.MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)
        self.ffn = keras.Sequential([
            layers.Dense(ff_dim, activation='relu'),
            layers.Dense(key_dim)
        ])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, x, training):
        attn_output = self.mha(x, x)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)


def build_transformer_model(vocab_size, d_model, num_heads, ff_dim, maxlen):
    inputs = layers.Input(shape=(maxlen,))
    embedding_layer = layers.Embedding(input_dim=vocab_size, output_dim=d_model)
    x = embedding_layer(inputs)

    transformer_block = TransformerEncoder(num_heads=num_heads, key_dim=d_model, ff_dim=ff_dim)
    x = transformer_block(x, training=True)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.1)(x)
    outputs = layers.Dense(vocab_size, activation="softmax")(x)

    return models.Model(inputs=inputs, outputs=outputs)

def train(x, y, tokens, tokens_indices, epochs=200):
    x = np.asarray(x)
    y = np.asarray(y)
    dataset = tf.data.Dataset.from_tensor_slices((x, y))
    dataset = dataset.shuffle(buffer_size=4096)
    dataset = dataset.batch(128)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    maxlen = x.shape[1]
    model = build_transformer_model(vocab_size=len(tokens), d_model=256, num_heads=8, ff_dim=512, maxlen=maxlen)

    optimizer = optimizers.Adam(learning_rate=0.001)
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
                    next_index = sample(preds, temperature=temperature)
                    next_token = tokens[next_index]
                    print(next_token, end='')


                    text_cut = text_cut[1: 60] + [next_token]
                print('end')


if __name__ == '__main__':
    file = DATA_PATH + '越女剑.txt'
    d = get_single_corpus(file)
    _x, _y, _tokens, _tokens_indices = get_dataset(d)
    train(_x, _y, _tokens, _tokens_indices)
