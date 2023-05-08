import pandas as pd
import numpy as np
from ast import literal_eval
from keras.layers import Input, Embedding, LSTM, Bidirectional, Dense, Dot, Concatenate
from keras.models import Model
from gensim.models import Word2Vec
import tensorflow as tf

# load data
data = pd.read_csv('match_data1.csv').reset_index(drop=True)
data = data.replace("[]", np.nan)

data = data.dropna(how='any')
data = data.reset_index(drop=True)

# load word2vec model
model = Word2Vec.load('word2vec.model1')
hero_vec = {hero: model.wv[hero] for hero in model.wv.index_to_key}

#get data shape
n_data = len(data)
vocab_size = len(hero_vec)


X0 = np.empty((n_data, 5, 150))
X1 = np.empty((n_data, 5, 150))
y = np.empty(n_data, dtype=np.int32)

counter_matrix0 = np.empty((n_data, 5,5))
counter_matrix1 = np.empty((n_data, 5,5))
# counnter_matrix
df_counter_rate = pd.read_csv('counter_rate1.csv', index_col=0, header=0)
def get_counter_rate(hero1, hero2):
    return df_counter_rate.loc[hero1, hero2].astype(np.float32)

# print(type(get_counter_rate('Axe', 'Anti-Mage')))


for i, row in data.iterrows():
    # get hero picks and pick order from data
    radiant_picks = literal_eval(row['radiant_pick'])
    dire_picks = literal_eval(row['dire_pick'])
    radiant_pick_order = literal_eval(row['radiant_pick_order'])
    dire_pick_order = literal_eval(row['dire_pick_order'])
    counter0 = []
    counter1 = []
    # print(heroes)
    if len(radiant_picks)==5 and len(dire_picks) == 5:
        key0 = str(radiant_pick_order)
        hero_items0 = list(zip(radiant_picks, radiant_pick_order))
        sorted_heroes0 = [x[0] for x in sorted(hero_items0, key=lambda x: x[1], reverse=False)]
        key1 = str(dire_pick_order)
        hero_items1 = list(zip(dire_picks, dire_pick_order))
        sorted_heroes1 = [x[0] for x in sorted(hero_items1, key=lambda x: x[1], reverse=False)]
        for hero1 in sorted_heroes0:
            for hero2 in sorted_heroes1:
                counter0.append(get_counter_rate(hero1,hero2))
        counter0 = np.array(counter0)

        for hero1 in sorted_heroes1:
            for hero2 in sorted_heroes0:
                counter1.append(get_counter_rate(hero1,hero2))
        counter1 = np.array(counter1)
        counter_matrix0[i, :, :] = counter0.reshape(5,5)
        counter_matrix1[i, :, :] = counter1.reshape(5,5)
        # print(sorted_heroes)
        radiant_picks = np.array([hero_vec[hero] for hero in sorted_heroes0])
        dire_picks = np.array([hero_vec[hero] for hero in sorted_heroes1])
        X0[i, :, :] = radiant_picks
        X1[i, :, :] = dire_picks
        print(i)
        # get match result
        y[i] = int(row['radiant_win'])
        #如果y中的值不是0或1，删除所有包含这些值的行
for i in reversed(range(len(y))):
    if y[i] != 0 and y[i] != 1:
        X0 = np.delete(X0, i, axis=0)
        X1 = np.delete(X1, i, axis=0)
        y = np.delete(y, i, axis=0)
        counter_matrix0 = np.delete(counter_matrix0, i, axis=0)
        counter_matrix1 = np.delete(counter_matrix1, i, axis=0)
        print(i)

        
df = pd.DataFrame(y)
df.to_csv('y.csv')


print(X0.shape, X1.shape, y.shape, counter_matrix0.shape, counter_matrix1.shape)
np.savez('data1.npz',X0=X0, X1=X1, y=y,counter0=counter_matrix0,counter1=counter_matrix1)

            

