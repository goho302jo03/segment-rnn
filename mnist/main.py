import random
import sys
import time
import numpy as np
import tensorflow as tf
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Activation, Dense
from keras.layers.recurrent import LSTM
from keras.callbacks import EarlyStopping
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split

np.random.seed(0)
random.seed(0)
tf.set_random_seed(0)

(tr_x, tr_y), (te_x, te_y) = mnist.load_data()
nb_classes = 10
img_rows, img_cols = 28, 28

tr_x = tr_x.astype('float32')
te_x = te_x.astype('float32')
tr_x /= 255
te_x /= 255

tr_y = np_utils.to_categorical(tr_y, nb_classes)
te_y = np_utils.to_categorical(te_y, nb_classes)

tr_x, va_x, tr_y, va_y = train_test_split(tr_x, tr_y, test_size=0.2, random_state=int(sys.argv[5]))

mode = sys.argv[1]
if mode == 'full':
    SEGMENT_LENGTH = img_rows

elif mode == 'segment':
    SAMPLE_NUM = int(sys.argv[2])
    SEGMENT_LENGTH = int(sys.argv[3])
    BORDER_LENGTH = int(sys.argv[4])
    tmp_x, tmp_y = [], []
    for i in range(len(tr_x)):
        start_idx = random.sample(list(np.arange(BORDER_LENGTH, img_rows-SEGMENT_LENGTH+1-BORDER_LENGTH)), k=SAMPLE_NUM)
        for idx in start_idx:
            tmp_x.append(tr_x[i][idx:idx+SEGMENT_LENGTH])
            tmp_y.append(tr_y[i])

    tr_x = np.asarray(tmp_x)
    tr_y = np.asarray(tmp_y)

print(f'tr_x shape: {np.shape(tr_x)}')
print(f'va_x shape: {np.shape(va_x)}')
print(f'te_x shape: {np.shape(te_x)}')

nb_units = 50

model = Sequential()
model.add(LSTM(nb_units, input_shape=(None, img_cols)))
model.add(Dense(units=nb_classes, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

BATCH_SIZE = 128
NUM_EPOCHS = 1000

es = EarlyStopping(patience=10, restore_best_weights=True)
model.summary()
start = time.time()
model.fit(tr_x, tr_y, batch_size=BATCH_SIZE, epochs=NUM_EPOCHS, validation_data=(va_x, va_y), callbacks=[es], verbose=1)
end = time.time()


result = model.evaluate(te_x, te_y)
print(f'Test acc: {result[1]}')
print(f'Test auc: {roc_auc_score(te_y, model.predict(te_x))}')
print(f'Execute time: {end-start}')
