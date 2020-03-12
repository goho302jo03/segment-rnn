import random
import sys
import os
import time
import numpy as np
import tensorflow as tf
from keras.preprocessing import sequence
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Activation, Dense
from keras.layers.recurrent import LSTM
from keras.layers import GRU, SimpleRNN
from keras.callbacks import EarlyStopping
from keras.layers.embeddings import Embedding
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score, matthews_corrcoef
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from keras.models import load_model
from keras.utils import to_categorical

sys.path.append('../')
from utils.core import Pipeline


class Amp(Pipeline):
    def __init__(self):
        super(Amp, self).__init__()
        self.nb_classes = 10
        self.nb_units = 64
        self.embedding_size = 128
        self.batch_size = 32
        self.mode = sys.argv[2]
        self.neuron_type = LSTM
        self.resample = False if 'full' == self.mode else bool(int(sys.argv[3]))
        self.segment_len = 200 if 'full' == self.mode else int(sys.argv[4])
        self.sample_num = 1 if 'full' == self.mode or self.resample else int(sys.argv[5])


    def data_loader(self):
        tr_x = np.load('./data/tr_x.npy', allow_pickle=True)
        tr_y = np.load('./data/tr_y.npy', allow_pickle=True)
        va_x = np.load('./data/va_x.npy', allow_pickle=True)
        va_y = np.load('./data/va_y.npy', allow_pickle=True)
        te_x = np.load('./data/te_x.npy', allow_pickle=True)
        te_y = np.load('./data/te_y.npy', allow_pickle=True)
        tr_seq_len = [len(v) for v in tr_x]
        tr_x = sequence.pad_sequences(tr_x, maxlen=200, padding='pre')
        tr_x = to_categorical(tr_x)
        va_x = to_categorical(va_x)
        te_x = to_categorical(te_x)
        return tr_x, tr_y, va_x, va_y, te_x, te_y, tr_seq_len


    def sample_segment(self, x, x_seq_len, y):
        tmp_x, tmp_y = [], []
        for i in range(len(x)):
            all_in_num = max(x_seq_len[i]-self.segment_len+1, 0)
            all_in_choose_num = min(all_in_num, self.sample_num)
            all_in_start_idx = random.sample(list(np.arange(all_in_num)), k=all_in_choose_num)
            shift = 200-x_seq_len[i]
            for idx in all_in_start_idx:
                tmp_x.append(x[i][shift+idx:shift+idx+self.segment_len])
                tmp_y.append(y[i])

            res_num = min(self.sample_num-all_in_choose_num, self.segment_len-1, x_seq_len[i])
            for idx in range(res_num):
                tmp_x.append(x[i][200-idx-self.segment_len-all_in_num:200-idx-all_in_num])
                tmp_y.append(y[i])
        return np.asarray(tmp_x), np.asarray(tmp_y)


    def build_model(self):
        model = Sequential()
        #model.add(Embedding(21, self.embedding_size))
        model.add(self.neuron_type(self.nb_units, dropout=0.2, input_shape=(None, 21), recurrent_dropout=0.2, return_sequences=True))
        model.add(self.neuron_type(self.nb_units, dropout=0.2, recurrent_dropout=0.2))
        model.add(Dense(units=32, activation='relu'))
        model.add(Dense(units=32, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))

        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.summary()
        return model

if '__main__' == __name__:
    pipeline = Amp()
    tr_x, tr_y, va_x, va_y, te_x, te_y, x_seq_len = pipeline.data_loader()
    exec_time, n_epoch, exec_time_without_calc = pipeline.run(tr_x, tr_y, va_x, va_y, x_seq_len)

    model = load_model('./tmp/model.h5')
    result = model.evaluate(te_x, te_y)
    print(f'Test acc: {result[1]}')
    print(f'Test auc: {roc_auc_score(te_y, model.predict(te_x))}')
    print(f'Test mcc: {matthews_corrcoef(te_y, [1 if i>0.5 else 0 for i in model.predict(te_x)])}')
    print(f'Execute epoch: {n_epoch}')
    print(f'Execute time: {exec_time}')
    if pipeline.resample:
        print(f'(W) Execute time: {exec_time_without_calc}')
