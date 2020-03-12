import numpy as np
import tensorflow as tf
import nltk
import random
import sys
import os
import time
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
from keras.models import load_model
import nltk
nltk.download('punkt')
sys.path.append('../')
from utils.core import Pipeline


class Sentiment(Pipeline):
    def __init__(self):
        super(Sentiment, self).__init__()
        self.nb_classes = 10
        self.nb_units = 64
        self.embedding_size = 128
        self.batch_size = 32
        self.max_len = 0
        self.vocab_size = 0
        self.mode = sys.argv[2]
        self.neuron_type = LSTM
        self.resample = False if 'full' == self.mode else bool(int(sys.argv[3]))
        self.segment_len = 9999 if 'full' == self.mode else int(sys.argv[4])
        self.sample_num = 1 if 'full' == self.mode or self.resample else int(sys.argv[5])


    def data_loader(self):
        with open('./data/training.txt', 'r') as f:
            data = [v.rstrip('\n').split('\t') for v in f.readlines()]

        word_freq = dict()
        data_num = len(data)
        for row in data:
            words = nltk.word_tokenize(row[1].strip().lower())
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
        y = np.zeros(data_num)
        for i, row in enumerate(data):
            label, sentence = row
            words = nltk.word_tokenize(sentence.strip().lower())
            seqs = []
            for word in words:
                if word in word2index:
                    seqs.append(word2index[word])
                else:
                    seqs.append(word2index["UNK"])
            x[i] = seqs
            y[i] = int(label)

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
            all_in_num = max(x_seq_len[i]-self.segment_len+1, 0)
            all_in_choose_num = min(all_in_num, self.sample_num)
            all_in_start_idx = random.sample(list(np.arange(all_in_num)), k=all_in_choose_num)
            shift = self.max_len-x_seq_len[i]
            for idx in all_in_start_idx:
                tmp_x.append(x[i][shift+idx:shift+idx+self.segment_len])
                tmp_y.append(y[i])

            res_num = min(self.sample_num-all_in_choose_num, self.segment_len-1, x_seq_len[i])
            for idx in range(res_num):
                tmp_x.append(x[i][self.max_len-idx-self.segment_len-all_in_num:self.max_len-idx-all_in_num])
                tmp_y.append(y[i])
        return np.asarray(tmp_x), np.asarray(tmp_y)


    def build_model(self):
        model = Sequential()
        model.add(Embedding(self.vocab_size, self.embedding_size))
        model.add(self.neuron_type(self.nb_units, activation='relu', dropout=0.2, recurrent_dropout=0.2, return_sequences=True))
        model.add(self.neuron_type(self.nb_units, activation='relu', dropout=0.2, recurrent_dropout=0.2, return_sequences=True))
        model.add(self.neuron_type(self.nb_units, activation='relu', dropout=0.2, recurrent_dropout=0.2))
        model.add(Dense(units=16, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))

        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.summary()
        return model

if '__main__' == __name__:
    pipeline = Sentiment()
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
