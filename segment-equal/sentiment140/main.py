import numpy as np
import tensorflow as tf
import nltk
import random
import sys
import os
import time
import collections
from math import floor
from keras.layers.core import Activation, Dense
from keras.preprocessing import sequence
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.layers import GRU, SimpleRNN
from keras.callbacks import EarlyStopping
from keras.models import Sequential
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from keras.utils import np_utils
from keras.models import load_model
sys.path.append('../')
from utils.core import Pipeline


class Sentiment140(Pipeline):
    def __init__(self):
        super(Sentiment140, self).__init__()
        self.nb_classes = 10
        self.nb_units = 64
        self.embedding_size = 128
        self.batch_size = 512
        self.max_len = 0
        self.vocab_size = 0
        self.mode = sys.argv[2]
        self.neuron_type = LSTM
        self.resample = False if 'full' == self.mode else bool(int(sys.argv[3]))
        self.segment_len = 9999 if 'full' == self.mode else int(sys.argv[4])
        self.sample_num = 1 if 'full' == self.mode or self.resample else int(sys.argv[5])


    def data_loader(self):
        with open('./data/training.1600000.processed.noemoticon.csv', 'r', encoding='ISO-8859-1') as f:
            data = f.readlines()
            data = [data[i].rstrip('\n').split(',', 5) for i in range(0, len(data), 20)]
        y = np.array([int(v[0].strip('"')) for v in data])
        y = np.where(y==4, 1, y)
        data = [v[-1].strip('"').replace('.', '').replace(',', '') for v in data]

        word_freq = collections.Counter()
        data_num = len(data)
        for row in data:
            words = nltk.word_tokenize(row.strip().lower())
            self.max_len = max(self.max_len, len(words))
            for word in words:
                if word not in word_freq:
                    word_freq[word] = 0
                word_freq[word] += 1

        self.segment_len = min(self.segment_len, self.max_len)
        self.vocab_size = len(word_freq) + 2
        word2index = {x: i+2 for i, x in enumerate(word_freq)}
        word2index["PAD"] = 0
        word2index["UNK"] = 1
        index2word = {v:k for k, v in word2index.items()}

        x = np.empty(data_num, dtype=list)
        for i, sentence in enumerate(data):
            words = nltk.word_tokenize(sentence.strip().lower())
            seqs = []
            for word in words:
                if word in word2index:
                    seqs.append(word2index[word])
                else:
                    seqs.append(word2index["UNK"])
            x[i] = seqs

        y = np_utils.to_categorical(y, 2)

        tmp_x, te_x, tmp_y, te_y = train_test_split(x, y, test_size=0.2, random_state=int(sys.argv[1]))
        tr_x, va_x, tr_y, va_y = train_test_split(tmp_x, tmp_y, test_size=0.2, random_state=int(sys.argv[1]))
        tr_seq_len = [len(v) for v in tr_x]
        tr_x = sequence.pad_sequences(tr_x, maxlen=self.max_len, padding='pre')
        va_x = sequence.pad_sequences(va_x, maxlen=self.max_len, padding='pre')
        te_x = sequence.pad_sequences(te_x, maxlen=self.max_len, padding='pre')
        return tr_x, tr_y, va_x, va_y, te_x, te_y, tr_seq_len


    def sample_segment(self, x, x_seq_len, y):
        tmp_x, tmp_y = [], []
        for i in range(len(x)):
            n_segment = x_seq_len[i] // self.segment_len
            shift = self.max_len-x_seq_len[i]
            for idx in range(n_segment):
                tmp_x.append(x[i][shift+idx*self.segment_len:shift+(idx+1)*self.segment_len])
                tmp_y.append(y[i])
            tmp_x.append(x[i][shift+(n_segment)*self.segment_len:])
            tmp_y.append(y[i])
        tmp_x = sequence.pad_sequences(tmp_x, maxlen=self.segment_len, padding='pre')
        return np.asarray(tmp_x), np.asarray(tmp_y)


    def build_model(self):
        model = Sequential()
        model.add(Embedding(self.vocab_size, self.embedding_size))
        model.add(self.neuron_type(self.nb_units, activation='relu', dropout=0.1, recurrent_dropout=0.1, return_sequences=True))
        model.add(self.neuron_type(self.nb_units, activation='relu', dropout=0.1, recurrent_dropout=0.1, return_sequences=True))
        model.add(self.neuron_type(self.nb_units, activation='relu', dropout=0.1, recurrent_dropout=0.1, return_sequences=True))
        model.add(self.neuron_type(self.nb_units, activation='relu', dropout=0.1, recurrent_dropout=0.1))
        model.add(Dense(2, activation='softmax'))

        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.summary()
        return model

if '__main__' == __name__:
    pipeline = Sentiment140()
    tr_x, tr_y, va_x, va_y, te_x, te_y, x_seq_len = pipeline.data_loader()
    exec_time, n_epoch, exec_time_without_calc = pipeline.run(tr_x, tr_y, va_x, va_y, x_seq_len)

    model = load_model('./tmp/model.h5')
    result = model.evaluate(te_x, te_y)
    print(f'Test acc: {result[1]}')
    # print(f'Test auc: {roc_auc_score(te_y, model.predict(te_x))}')
    print(f'Execute epoch: {n_epoch}')
    print(f'Execute time: {exec_time}')
    if pipeline.resample:
        print(f'(W) Execute time: {exec_time_without_calc}')
