import random
import sys
import os
import time
import numpy as np
import tensorflow as tf
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Activation, Dense
from keras.layers.recurrent import LSTM
from keras.layers import GRU, SimpleRNN
from keras.callbacks import EarlyStopping
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from keras.models import load_model
sys.path.append('../')
from utils.core import Pipeline


class Fashion(Pipeline):
    def __init__(self):
        super(Fashion, self).__init__()
        self.nb_classes = 10
        self.nb_units = 50
        self.img_rows, self.img_cols = 28, 28
        self.batch_size = 128
        self.mode = sys.argv[2]
        self.neuron_type = LSTM
        self.resample = False if 'full' == self.mode else bool(int(sys.argv[3]))
        self.segment_len = self.img_rows if 'full' == self.mode else int(sys.argv[4])
        self.sample_num = 1 if 'full' == self.mode or self.resample else int(sys.argv[5])


    def data_loader(self):
        with open('./data/fashion-mnist_train.csv', 'r') as f:
            tr = np.asarray([v.rstrip('\n').split(',') for v in f.readlines()[1:]])
        with open('./data/fashion-mnist_test.csv', 'r') as f:
            te = np.asarray([v.rstrip('\n').split(',') for v in f.readlines()[1:]])
        tr_x, tr_y = np.reshape(tr[:, 1:], (-1, 28, 28)), tr[:, 0]
        te_x, te_y = np.reshape(te[:, 1:], (-1, 28, 28)), te[:, 0]
        tr_x = tr_x.astype('float32') / 255
        te_x = te_x.astype('float32') / 255
        tr_y = np_utils.to_categorical(tr_y, self.nb_classes)
        te_y = np_utils.to_categorical(te_y, self.nb_classes)
        tr_x, va_x, tr_y, va_y = train_test_split(tr_x, tr_y, test_size=0.2, random_state=int(sys.argv[1]))
        print(f'tr_x shape: {np.shape(tr_x)}')
        print(f'va_x shape: {np.shape(va_x)}')
        print(f'te_x shape: {np.shape(te_x)}')
        return tr_x, tr_y, va_x, va_y, te_x, te_y, None


    def sample_segment(self, x, x_seq_len, y):
        tmp_x, tmp_y = [], []
        for i in range(len(x)):
            start_idx = random.sample(list(np.arange(self.img_rows-self.segment_len+1)), k=self.sample_num)
            for idx in start_idx:
                tmp_x.append(x[i][idx:idx+self.segment_len])
                tmp_y.append(y[i])
        return np.asarray(tmp_x), np.asarray(tmp_y)


    def build_model(self):
        model = Sequential()
        model.add(self.neuron_type(self.nb_units, input_shape=(None, self.img_cols), return_sequences=True))
        model.add(self.neuron_type(self.nb_units))
        model.add(Dense(units=16, activation='relu'))
        model.add(Dense(units=self.nb_classes, activation='softmax'))

        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.summary()
        return model

if '__main__' == __name__:
    pipeline = Fashion()
    tr_x, tr_y, va_x, va_y, te_x, te_y, _ = pipeline.data_loader()
    exec_time, n_epoch, exec_time_without_calc = pipeline.run(tr_x, tr_y, va_x, va_y, None)

    model = load_model('./tmp/model.h5')
    result = model.evaluate(te_x, te_y)
    print(f'Test acc: {result[1]}')
    print(f'Test auc: {roc_auc_score(te_y, model.predict(te_x))}')
    print(f'Execute epoch: {n_epoch}')
    print(f'Execute time: {exec_time}')
    if pipeline.resample:
        print(f'(W) Execute time: {exec_time_without_calc}')
