import pandas as pd
import numpy as np
from keras.layers import Input, Embedding, LSTM, Bidirectional, Dense, Dot, Concatenate, Multiply, Dropout
from keras.utils.np_utils import normalize
from keras.models import Model
from gensim.models import Word2Vec
from keras import regularizers
import keras
from sklearn.model_selection import StratifiedKFold
import tensorflow as tf
tf.config.list_physical_devices('GPU')
# load data
data = np.load('data.npz')

# get data
X = data['X']
y = data['y']
counter_matrix = data['counter_matrix']
counter_matrix = normalize(counter_matrix, axis=-1)
counter_matrix = np.mean(counter_matrix, axis=-1, keepdims=True)
print(X.shape)
print(y.shape)
print(counter_matrix.shape)

# split data 10-fold
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
for train_index, test_index in skf.split(X, y):
    X_train, X_test = X[train_index], X[test_index]
    counter_matrix_train, counter_matrix_test = counter_matrix[train_index], counter_matrix[test_index]
    y_train, y_test = y[train_index], y[test_index]

input_layer = Input(shape=(10,150))
attention_matrix = Input(shape=(10,1))
print('input_layer:',input_layer.shape, 'attention_matrix:',attention_matrix.shape)



# attention
# attention_layer = Multiply()([input_layer, attention_matrix])
# print('attention_layer:',attention_layer.shape)

# LSTM
lstm_layer = LSTM(128, return_sequences=True,kernel_regularizer=regularizers.l2(0.025))(input_layer)
print('lstm_layer:',lstm_layer.shape)

# attention层

attention_output = Multiply()([lstm_layer, attention_matrix])
print('attention_output:',attention_output.shape)

# concat_layer
concat_layer = keras.layers.concatenate([attention_output,lstm_layer], axis=-1)
print('concat_layer:',concat_layer.shape)


# lstm_layer_output = LSTM(32, return_sequences=True)(concat_layer)

dense_output = Dense(64, activation='relu')(concat_layer)
dropout_output = Dropout(0.2)(dense_output)
flatten_layer = keras.layers.Flatten()(dropout_output)
output_layer = Dense(1, activation='sigmoid')(flatten_layer)


# model
model = Model(inputs=[input_layer, attention_matrix], outputs=output_layer)
optimizer = keras.optimizers.Adam(learning_rate=0.00001)
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'],loss_weights=[0.9, 0.0])

# train
model.fit([X_train, counter_matrix_train], y_train, epochs=10000, batch_size=2000, validation_data=([X_test, counter_matrix_test], y_test))

# evaluate
model.evaluate([X_train, counter_matrix_train], y_train)