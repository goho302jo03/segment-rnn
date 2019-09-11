import random
import sys
import time
import numpy as np
import tensorflow as tf
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Activation, Dense
from keras.layers.recurrent import LSTM
from keras.layers import GRU
from keras.callbacks import EarlyStopping
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split

np.random.seed(0)
random.seed(0)
tf.set_random_seed(0)

nb_classes = 10
nb_units = 50
img_rows, img_cols = 28, 28
batch_size = 128
epochs = 1000

mode = sys.argv[2]
neuron_type = LSTM
resample = False if 'full' == mode else bool(int(sys.argv[3]))
segment_len = img_rows if 'full' == mode else int(sys.argv[4])
sample_num = 1 if 'full' == mode or resample else int(sys.argv[5])

def data_loader():
    with open('./data/fashion-mnist_train.csv', 'r') as f:
        tr = np.asarray([v.rstrip('\n').split(',') for v in f.readlines()[1:]])
    with open('./data/fashion-mnist_test.csv', 'r') as f:
        te = np.asarray([v.rstrip('\n').split(',') for v in f.readlines()[1:]])
    tr_x, tr_y = np.reshape(tr[:, 1:], (-1, 28, 28)), tr[:, 0]
    te_x, te_y = np.reshape(te[:, 1:], (-1, 28, 28)), te[:, 0]
    tr_x = tr_x.astype('float32') / 255
    te_x = te_x.astype('float32') / 255
    tr_y = np_utils.to_categorical(tr_y, nb_classes)
    te_y = np_utils.to_categorical(te_y, nb_classes)
    tr_x, va_x, tr_y, va_y = train_test_split(tr_x, tr_y, test_size=0.2, random_state=int(sys.argv[1]))
    print(f'tr_x shape: {np.shape(tr_x)}')
    print(f'va_x shape: {np.shape(va_x)}')
    print(f'te_x shape: {np.shape(te_x)}')
    return tr_x, tr_y, va_x, va_y, te_x, te_y


def sample_segment(x, y):
    tmp_x, tmp_y = [], []
    for i in range(len(x)):
        start_idx = random.sample(list(np.arange(img_rows-segment_len+1)), k=sample_num)
        for idx in start_idx:
            tmp_x.append(x[i][idx:idx+segment_len])
            tmp_y.append(y[i])
    return np.asarray(tmp_x), np.asarray(tmp_y)


def build_model():
    model = Sequential()
    model.add(neuron_type(nb_units, input_shape=(None, img_cols), return_sequences=True))
    model.add(neuron_type(nb_units))
    model.add(Dense(units=16, activation='relu'))
    model.add(Dense(units=nb_classes, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    return model


def non_resample_train(model, tr_x, tr_y, va_x, va_y):
    es = EarlyStopping(patience=10, restore_best_weights=True)
    start = time.time()
    history = model.fit(tr_x, tr_y, batch_size=batch_size, epochs=epochs, validation_data=(va_x, va_y), callbacks=[es], verbose=1)
    end = time.time()
    return model, end-start, len(history.epoch)


def resample_train(model, tr_x, tr_y, va_x, va_y):
    es = EarlyStopping(patience=10, restore_best_weights=True)
    patience = 10
    best_val_loss = 9999
    best_model = None
    exec_time_without_calc = 0
    start = time.time()
    for i in range(epochs):
        e_tr_x, e_tr_y = sample_segment(tr_x, tr_y)
        w_start = time.time()
        history = model.fit(e_tr_x, e_tr_y, batch_size=batch_size, epochs=1, validation_data=(va_x, va_y))
        exec_time_without_calc += time.time() - w_start
        if history.history['val_loss'][0] < best_val_loss:
            best_val_loss = history.history['val_loss']
            best_epochs_count = 0
            best_model = model
        else:
            best_epochs_count += 1

        if patience == best_epochs_count:
            n_epoch = i
            break
    end = time.time()
    return best_model, end-start, exec_time_without_calc, n_epoch


def main():
    tr_x, tr_y, va_x, va_y, te_x, te_y = data_loader()
    model = build_model()

    if resample:
        model, exec_time, exec_time_without_calc, n_epoch = resample_train(model, tr_x, tr_y, va_x, va_y)
    else:
        tr_x, tr_y = sample_segment(tr_x, tr_y)
        model, exec_time, n_epoch = non_resample_train(model, tr_x, tr_y, va_x, va_y)

    result = model.evaluate(te_x, te_y)
    print(f'Test acc: {result[1]}')
    print(f'Test auc: {roc_auc_score(te_y, model.predict(te_x))}')
    print(f'Execute epoch: {n_epoch}')
    print(f'Execute time: {exec_time}')
    if resample:
        print(f'(W) Execute time: {exec_time_without_calc}')


if '__main__' == __name__:
    main()
