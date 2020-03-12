import numpy as np
from sklearn.preprocessing import LabelEncoder

def del_padding(file):
    x = np.load(file)
    tmp_x = []
    for i in range(len(x)):
        tmp_x.append(x[i][np.count_nonzero(x[i]==0):])
    return tmp_x

tr_x = del_padding('./data/tr_x.npy')
np.save('./data/tr_x.npy', tr_x)
