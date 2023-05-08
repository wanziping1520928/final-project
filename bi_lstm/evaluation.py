import numpy as np
from keras.saving.save import load_model
from keras.utils import normalize
import pandas as pd
from sklearn.model_selection import StratifiedKFold

data = np.load('data1.npz')
# get data
X0 = normalize(data['X0'],axis=-1)
X1 = normalize(data['X1'],axis=-1)
y = data['y']
counter_rate0 = data['counter0']
counter_rate1 = data['counter1']
counter_n0 = normalize(counter_rate0, axis=-1)
counter0 = np.mean(counter_n0, axis=-1, keepdims=True)
counter_n1 = normalize(counter_rate1, axis=-1)
counter1 = np.mean(counter_n1, axis=-1, keepdims=True)



# print(counter_n0)
# print(X0)
indices = np.arange(len(X0))
np.random.shuffle(indices)

split = int(0.2 * len(X0))

X0_test = X0[indices[:split]]
X1_test = X1[indices[:split]]
counter0_test = counter0[indices[:split]]
counter1_test = counter1[indices[:split]]
y_test = y[indices[:split]]
print(X0_test.shape,X1_test.shape,counter0_test.shape,counter1_test.shape)
model = load_model('bi_lstm.h5')

out0 = model.predict([X0_test,X1_test,counter0_test,counter1_test])
print(out0)
for i in range(5):
    tag_line = 0.6 + 0.05 * i
    correct_num = 0
    compare_num = 0
    for i in range(len(out0)):
        if out0[i][0] < (1.0 - tag_line) or out0[i][0] > tag_line:
            compare_num += 1
            if out0[i][0] < 0.5:
                temp_result = 0.0
            else:
                temp_result = 1.0
            if temp_result == y_test[i]:
                correct_num += 1
    if compare_num != 0:
        print('test set,win rate over' + str(tag_line) + 'accuracy：', float(correct_num) / compare_num, \
              ' (' + str(correct_num) + '/' + str(compare_num) + ')')
    else:
        print('test set,win rate over' + str(tag_line) + 'accuracy：', '0.0', \
              ' (' + str(correct_num) + '/' + str(compare_num) + ')')