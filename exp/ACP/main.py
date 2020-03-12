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
from keras.optimizers import Adam
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score, matthews_corrcoef
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from keras.models import load_model
from keras.utils import to_categorical
sys.path.append('../')
from utils.core import Pipeline


class ACP(Pipeline):
    def __init__(self):
        super(ACP, self).__init__()
        self.nb_classes = 1
        self.nb_units = 64
        self.embedding_size = 21
        self.batch_size = 32
        self.mode = sys.argv[2]
        self.patience = 20
        self.neuron_type = LSTM
        self.resample = False if 'full' == self.mode else bool(int(sys.argv[3]))
        self.segment_len = 97 if 'full' == self.mode else int(sys.argv[4])
        self.sample_num = 1 if 'full' == self.mode or self.resample else int(sys.argv[5])


    def data_loader(self):
        tmp_x = np.load('./data/tr_x.npy', allow_pickle=True)
        tmp_y = np.load('./data/tr_y.npy', allow_pickle=True)
        te_x = np.load('./data/te_x.npy', allow_pickle=True)
        te_y = np.load('./data/te_y.npy', allow_pickle=True)
        tr_x, va_x, tr_y, va_y = train_test_split(tmp_x, tmp_y, test_size=0.1, random_state=int(sys.argv[1]))
        tr_seq_len = [len(v) for v in tr_x]
        tr_x = sequence.pad_sequences(tr_x, maxlen=97, padding='pre')
        va_x = sequence.pad_sequences(va_x, maxlen=97, padding='pre')
        te_x = sequence.pad_sequences(te_x, maxlen=97, padding='pre')
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
            shift = 97-x_seq_len[i]
            for idx in all_in_start_idx:
                tmp_x.append(x[i][shift+idx:shift+idx+self.segment_len])
                tmp_y.append(y[i])

            res_num = min(self.sample_num-all_in_choose_num, self.segment_len-1, x_seq_len[i])
            for idx in range(res_num):
                tmp_x.append(x[i][97-idx-self.segment_len-all_in_num:97-idx-all_in_num])
                tmp_y.append(y[i])
        return np.asarray(tmp_x), np.asarray(tmp_y)


    def build_model(self):
        model = Sequential()
        model.add(self.neuron_type(self.nb_units, input_shape=(None, 21), return_sequences=True))
        model.add(self.neuron_type(self.nb_units))
        model.add(Dense(units=16, activation='relu'))
        model.add(Dense(units=32, activation='relu'))
        model.add(Dense(units=64, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))

        model.compile(loss='binary_crossentropy', optimizer=Adam(clipvalue=0.001, clipnorm=1.), metrics=['accuracy'])
        model.summary()
        return model

if '__main__' == __name__:
    pipeline = ACP()
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
