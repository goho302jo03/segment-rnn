import numpy as np
import tensorflow as tf
import nltk
import random
import sys
import os
import time
from math import floor
from keras.datasets import imdb
from keras.layers.core import Activation, Dense
from keras.preprocessing import sequence
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.layers import GRU, SimpleRNN
from keras.layers import Dropout
from keras.callbacks import EarlyStopping
from keras.models import Sequential
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from keras.utils import np_utils
from keras.models import load_model
sys.path.append('../')
from utils.core import Pipeline


class Imdb(Pipeline):
    def __init__(self):
        super(Imdb, self).__init__()
        self.nb_classes = 10
        self.nb_units = 128
        self.embedding_size = 128
        self.batch_size = 128
        self.max_len = 605
        self.vocab_size = 20000
        self.mode = sys.argv[2]
        self.neuron_type = LSTM
        self.resample = False if 'full' == self.mode else bool(int(sys.argv[3]))
        self.segment_len = 9999 if 'full' == self.mode else int(sys.argv[4])
        self.sample_num = 1 if 'full' == self.mode or self.resample else int(sys.argv[5])


    def data_loader(self):
        (tmp_x, tmp_y), (te_x, te_y) = imdb.load_data(num_words=self.vocab_size)
        segment_len = min(self.segment_len, self.max_len)
        tr_x, va_x, tr_y, va_y = train_test_split(tmp_x, tmp_y, test_size=0.2, random_state=int(sys.argv[1]))
        tr_seq_len = [min(len(v), self.max_len) for v in tr_x]

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
        # model.add(self.neuron_type(self.nb_units, activation='relu', dropout=0.1, recurrent_dropout=0.1, return_sequences=True))
        # model.add(self.neuron_type(self.nb_units, activation='relu', dropout=0.1, recurrent_dropout=0.1, return_sequences=True))
        model.add(self.neuron_type(self.nb_units))
        # model.add(Dense(16, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.summary()
        return model

if '__main__' == __name__:
    pipeline = Imdb()
    tr_x, tr_y, va_x, va_y, te_x, te_y, x_seq_len = pipeline.data_loader()
    exec_time, n_epoch, exec_time_without_calc = pipeline.run(tr_x, tr_y, va_x, va_y, x_seq_len)

    model = load_model('./tmp/model.h5')
    result = model.evaluate(te_x, te_y)
    print(f'Test acc: {result[1]}')
    print(f'Test auc: {roc_auc_score(te_y, model.predict(te_x))}')
    print(f'Execute epoch: {n_epoch}')
    print(f'Execute time: {exec_time}')
    if pipeline.resample:
        print(f'(W) Execute time: {exec_time_without_calc}')
