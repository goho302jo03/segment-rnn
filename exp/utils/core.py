import random
import sys
import os
import time
import numpy as np
import tensorflow as tf
from keras.datasets import mnist
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


class Pipeline:
    def __init__(self):
        np.random.seed(0)
        random.seed(0)
        tf.set_random_seed(0)
        self.epochs = 1000
        self.patience = 10


    def non_resample_train(self, model, tr_x, tr_y, va_x, va_y):
        es = EarlyStopping(patience=self.patience, restore_best_weights=True)
        start = time.time()
        history = model.fit(tr_x, tr_y, batch_size=self.batch_size,
                            epochs=self.epochs, validation_data=(va_x, va_y),
                            callbacks=[es], verbose=1)
        end = time.time()
        model.save('./tmp/model.h5')
        return end-start, len(history.epoch), None


    def resample_train(self, model, tr_x, tr_y, va_x, va_y, x_seq_len):
        best_val_loss = float('inf')
        best_epochs_count = 0
        exec_time_without_calc = 0
        n_epoch = 0
        start = time.time()
        for i in range(self.epochs):
            e_tr_x, e_tr_y = self.sample_segment(tr_x, x_seq_len, tr_y)
            w_start = time.time()
            history = model.fit(e_tr_x, e_tr_y, batch_size=self.batch_size,
                                epochs=1, validation_data=(va_x, va_y))
            exec_time_without_calc += time.time() - w_start
            if history.history['val_loss'][0] <= best_val_loss:
                best_val_loss = history.history['val_loss'][0]
                best_epochs_count = 0
                model.save('./tmp/model.h5')
            else:
                best_epochs_count += 1

            if self.patience == best_epochs_count:
                n_epoch = i
                break
        end = time.time()
        return end-start, n_epoch, exec_time_without_calc


    def run(self, tr_x, tr_y, va_x, va_y, x_seq_len):
        model = self.build_model()

        if self.resample:
            return self.resample_train(model, tr_x, tr_y, va_x, va_y, x_seq_len)
        else:
            tr_x, tr_y = self.sample_segment(tr_x, x_seq_len, tr_y)
            return self.non_resample_train(model, tr_x, tr_y, va_x, va_y)
