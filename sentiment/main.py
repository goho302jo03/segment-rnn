import numpy as np
import tensorflow as tf
import nltk
import random
import sys
import time
from math import floor
from keras.layers.core import Activation, Dense
from keras.preprocessing import sequence
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.layers import GRU
from keras.callbacks import EarlyStopping
from keras.models import Sequential
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from keras.models import load_model

np.random.seed(0)
random.seed(0)
tf.set_random_seed(0)

nb_classes = 10
nb_units = 64
embedding_size = 128
batch_size = 32
epochs = 1000
max_len = 0
vocab_size = 0

mode = sys.argv[2]
neuron_type = LSTM
resample = False if 'full' == mode else bool(int(sys.argv[3]))
segment_len = 9999 if 'full' == mode else int(sys.argv[4])
sample_num = 1 if 'full' == mode or resample else int(sys.argv[5])

def data_loader():
    global segment_len, max_len, vocab_size

    with open('./data/training.txt', 'r') as f:
        data = [v.rstrip('\n').split('\t') for v in f.readlines()]

    word_freq = dict()
    data_num = len(data)
    for row in data:
        words = nltk.word_tokenize(row[1].strip().lower())
        max_len = max(max_len, len(words))
        for word in words:
            if word not in word_freq:
                word_freq[word] = 0
            word_freq[word] += 1

    segment_len = min(segment_len, max_len)
    vocab_size = len(word_freq) + 2
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
    tr_x = sequence.pad_sequences(tr_x, maxlen=max_len, padding='pre')
    va_x = sequence.pad_sequences(va_x, maxlen=max_len, padding='pre')
    te_x = sequence.pad_sequences(te_x, maxlen=max_len, padding='pre')
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
        shift = max_len-x_seq_len[i]
        for idx in all_in_start_idx:
            # print(tr_x[i][shift+idx:shift+idx+SEGMENT_LENGTH])
            tmp_x.append(x[i][shift+idx:shift+idx+segment_len])
            tmp_y.append(y[i])

        res_num = min(sample_num-all_in_choose_num, segment_len-1, x_seq_len[i])
        # print('part in:')
        for idx in range(res_num):
            # print(tr_x[i][200-idx-SEGMENT_LENGTH-all_in_num:200-idx-all_in_num])
            tmp_x.append(x[i][max_len-idx-segment_len-all_in_num:max_len-idx-all_in_num])
            tmp_y.append(y[i])
    return np.asarray(tmp_x), np.asarray(tmp_y)


def build_model():
    model = Sequential()
    model.add(Embedding(vocab_size, embedding_size))
    model.add(neuron_type(nb_units, activation='relu', dropout=0.2, recurrent_dropout=0.2, return_sequences=True))
    model.add(neuron_type(nb_units, activation='relu', dropout=0.2, recurrent_dropout=0.2, return_sequences=True))
    model.add(neuron_type(nb_units, activation='relu', dropout=0.2, recurrent_dropout=0.2))
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
    print(f'Execute epoch: {n_epoch}')
    print(f'Execute time: {exec_time}')
    if resample:
        print(f'(W) Execute time: {exec_time_without_calc}')


if '__main__' == __name__:
    main()
