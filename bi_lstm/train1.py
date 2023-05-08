import pandas as pd
import numpy as np
from keras.layers import Input, LSTM, Bidirectional, Dense, Dot, Concatenate, Multiply, Dropout, Permute
from keras.utils.np_utils import normalize
from keras.models import Model
from gensim.models import Word2Vec
from keras import regularizers
import keras
from sklearn.model_selection import StratifiedKFold
import tensorflow as tf
import keras.backend as K
tf.config.list_physical_devices('GPU')
# load data
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
print(X0.shape,X1.shape)


# print(counter_n0)
# print(X0)
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
for train_index,test_index in skf.split(X0, y):
    X0_train, X0_test = X0[train_index], X0[test_index]
    X1_train, X1_test = X1[train_index], X1[test_index]
    counter0_train, counter0_test = counter0[train_index], counter0[test_index]
    counter1_train, counter1_test = counter1[train_index], counter1[test_index]
    y_train, y_test = y[train_index], y[test_index]

def attention_3d_block(inputs):
    # inputs.shape = (batch_size, time_steps, input_dim)
    TIME_STEPS = int(inputs.shape[1])
    a = Permute((2, 1))(inputs)
    a = Dense(TIME_STEPS, activation='sigmoid')(a)
    a_probs = Permute((2, 1))(a)
    output_attention_mul = Multiply()([inputs, a_probs])
    return output_attention_mul


input_layer0 = Input(shape=(5,150))
input_layer1 = Input(shape=(5,150))
input_counter0 = Input(shape=(5,1))
input_counter1 = Input(shape=(5,1))

lstm_layer0 = LSTM(150,return_sequences=True)(input_layer0)
lstm_layer1 = LSTM(150,return_sequences=True)(input_layer1)

attention_mul0 = attention_3d_block(lstm_layer0)
attention_mul1 = attention_3d_block(lstm_layer1)
# print(attention_mul0.shape)

attention_output0 = Multiply()([attention_mul0,input_counter0])
attention_output1 = Multiply()([attention_mul1,input_counter1])
print(attention_output0.shape)
concat_layer = keras.layers.concatenate([attention_output0,attention_output1],axis=-1)
print(concat_layer.shape)
# print(concat_layer.shape)

dense_layer = Dense(128, activation='relu')(concat_layer)
dropout_layer = Dropout(0.2)(dense_layer)
flatten_layer = keras.layers.Flatten()(dropout_layer)
output_layer = Dense(1,activation='sigmoid')(flatten_layer)

#
model = Model(inputs=[input_layer0,input_layer1,input_counter0,input_counter1], outputs=output_layer)
optimizer = keras.optimizers.Adam(learning_rate=0.0001)
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

# train
callbacks = [keras.callbacks.EarlyStopping(monitor='val_loss', patience=20,mode='auto'), \
                 keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, mode='auto',cooldown=2, min_lr=0)]
model.fit([X0_train,X1_train,counter0_train,counter1_train], y_train, epochs=500, batch_size=2000,\
          validation_data=([X0_test,X1_test,counter0_test,counter1_test], y_test),callbacks=callbacks)
# evaluate

print(model.predict([X0_train,X1_train,counter0_train,counter1_train]),y_train)

model.evaluate([X0_test,X1_test,counter0_test,counter1_test], y_test)