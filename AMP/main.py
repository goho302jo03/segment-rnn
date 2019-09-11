import random
import sys
import time
import numpy as np
import tensorflow as tf
from keras.preprocessing import sequence
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Activation, Dense
from keras.layers.recurrent import LSTM
from keras.layers import GRU
from keras.callbacks import EarlyStopping
from keras.layers.embeddings import Embedding
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score, matthews_corrcoef
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from keras.models import load_model

np.random.seed(0)
random.seed(0)
tf.set_random_seed(0)

nb_classes = 10
nb_units = 64
embedding_size = 128
batch_size = 32
epochs = 1000

mode = sys.argv[2]
neuron_type = LSTM
resample = False if 'full' == mode else bool(int(sys.argv[3]))
segment_len = 200 if 'full' == mode else int(sys.argv[4])
sample_num = 1 if 'full' == mode or resample else int(sys.argv[5])

def data_loader():
    tr_x = np.load('./data/tr_x.npy', allow_pickle=True)
    tr_y = np.load('./data/tr_y.npy', allow_pickle=True)
    va_x = np.load('./data/va_x.npy', allow_pickle=True)
    va_y = np.load('./data/va_y.npy', allow_pickle=True)
    te_x = np.load('./data/te_x.npy', allow_pickle=True)
    te_y = np.load('./data/te_y.npy', allow_pickle=True)
    tr_seq_len = [len(v) for v in tr_x]
    tr_x = sequence.pad_sequences(tr_x, maxlen=200, padding='pre')
    return tr_x, tr_y, va_x, va_y, te_x, te_y, tr_seq_len


def sample_segment(x, x_seq_len, y):
    tmp_x, tmp_y = [], []
    for i in range(len(x)):
        # print('*'*20)
        # print(tr_x[i])
        all_in_num = max(x_seq_len[i]-segment_len+1, 0)
        all_in_choose_num = min(all_in_num, sample_num)
        all_in_start_idx = random.sample(list(np.arange(all_in_num)), k=all_in_choose_num)
        # print('all in:')
        shift = 200-x_seq_len[i]
        for idx in all_in_start_idx:
            # print(tr_x[i][shift+idx:shift+idx+SEGMENT_LENGTH])
            tmp_x.append(x[i][shift+idx:shift+idx+segment_len])
            tmp_y.append(y[i])

        res_num = min(sample_num-all_in_choose_num, segment_len-1, x_seq_len[i])
        # print('part in:')
        for idx in range(res_num):
            # print(tr_x[i][200-idx-SEGMENT_LENGTH-all_in_num:200-idx-all_in_num])
            tmp_x.append(x[i][200-idx-segment_len-all_in_num:200-idx-all_in_num])
            tmp_y.append(y[i])
    return np.asarray(tmp_x), np.asarray(tmp_y)


def build_model():
    model = Sequential()
    model.add(Embedding(21, embedding_size))
    model.add(neuron_type(nb_units, activation='relu', dropout=0.4, recurrent_dropout=0.4, return_sequences=True))
    model.add(neuron_type(nb_units, activation='relu', dropout=0.4, recurrent_dropout=0.4, return_sequences=True))
    model.add(neuron_type(nb_units, activation='relu', dropout=0.4, recurrent_dropout=0.4))
    model.add(Dense(units=16, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    return model


def non_resample_train(model, tr_x, tr_y, va_x, va_y):
    es = EarlyStopping(patience=10, restore_best_weights=True)
    start = time.time()
    history = model.fit(tr_x, tr_y, batch_size=batch_size, epochs=epochs, validation_data=(va_x, va_y), callbacks=[es], verbose=1)
    end = time.time()
    model.save('./tmp/model.h5')
    return end-start, len(history.epoch)


def resample_train(model, tr_seq_len, tr_x, tr_y, va_x, va_y):
    patience = 10
    best_val_loss = 9999
    exec_time_without_calc = 0
    start = time.time()
    for i in range(epochs):
        e_tr_x, e_tr_y = sample_segment(tr_x, tr_seq_len, tr_y)
        w_start = time.time()
        history = model.fit(e_tr_x, e_tr_y, batch_size=batch_size, epochs=1, validation_data=(va_x, va_y))
        exec_time_without_calc += time.time() - w_start
        if history.history['val_loss'][0] < best_val_loss:
            best_val_loss = history.history['val_loss']
            best_epochs_count = 0
            model.save('./tmp/model.h5')
        else:
            best_epochs_count += 1

        if patience == best_epochs_count:
            n_epoch = i
            break
    end = time.time()
    return end-start, exec_time_without_calc, n_epoch


def main():
    tr_x, tr_y, va_x, va_y, te_x, te_y, tr_seq_len = data_loader()
    model = build_model()

    if resample:
        exec_time, exec_time_without_calc, n_epoch = resample_train(model, tr_seq_len, tr_x, tr_y, va_x, va_y)
    else:
        tr_x, tr_y = sample_segment(tr_x, tr_seq_len, tr_y)
        exec_time, n_epoch = non_resample_train(model, tr_x, tr_y, va_x, va_y)

    model = load_model('./tmp/model.h5')
    result = model.evaluate(te_x, te_y)
    print(f'Test acc: {result[1]}')
    print(f'Test auc: {roc_auc_score(te_y, model.predict(te_x))}')
    print(f'Test mcc: {matthews_corrcoef(te_y, [1 if i>0.5 else 0 for i in model.predict(te_x)])}')
    print(f'Execute epoch: {n_epoch}')
    print(f'Execute time: {exec_time}')
    if resample:
        print(f'(W) Execute time: {exec_time_without_calc}')


if '__main__' == __name__:
    main()
