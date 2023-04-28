import pandas as pd
import numpy as np
from ast import literal_eval
from keras.layers import Input, Embedding, LSTM, Bidirectional, Dense, Dot, Concatenate
from keras.models import Model
from gensim.models import Word2Vec
import tensorflow as tf

# load data
data = pd.read_csv('match_data.csv').reset_index(drop=True)
data = data.replace("[]", np.nan)

data = data.dropna(how='any')
data = data.reset_index(drop=True)


# load word2vec model
model = Word2Vec.load('word2vec.model')
hero_vec = {hero: model.wv[hero] for hero in model.wv.index_to_key}

#get data shape
n_data = len(data)
vocab_size = len(hero_vec)


X = np.empty((n_data, 10, 150))
y = np.empty(n_data, dtype=np.int32)
counter_matrix = np.empty((n_data, 10, 10))
# counnter_matrix
df_counter_rate = pd.read_csv('counter_rate.csv', index_col=0, header=0)
def get_counter_rate(hero1, hero2):
    return df_counter_rate.loc[hero1, hero2].astype(np.float32)

print(type(get_counter_rate('Axe', 'Anti-Mage')))


for i, row in data.iterrows():
    # get hero picks and pick order from data
    radiant_picks = literal_eval(row['radiant_pick'])
    dire_picks = literal_eval(row['dire_pick'])
    radiant_pick_order = literal_eval(row['radiant_pick_order'])
    dire_pick_order = literal_eval(row['dire_pick_order'])
    heroes = radiant_picks + dire_picks
    order = radiant_pick_order + dire_pick_order
    # print(heroes)
    key = str(order)
    hero_items = list(zip(heroes, order))
    #ABBABAABAB
    #0123456789
    sorted_heroes = [x[0] for x in sorted(hero_items, key=lambda x: x[1], reverse=False)]

    counter_rate = np.array([[0,0,0,0,0,0,0,0,0,0],
                            [get_counter_rate(sorted_heroes[1],sorted_heroes[0])*1.1**1,0,0,0,0,0,0,0,0,0],
                            [get_counter_rate(sorted_heroes[2],sorted_heroes[0])*1.1**2,0,0,0,0,0,0,0,0,0],
                            [0,get_counter_rate(sorted_heroes[3],sorted_heroes[1])*1.1**3,get_counter_rate(sorted_heroes[3],sorted_heroes[2])*1.1**3,0,0,0,0,0,0,0],
                            [get_counter_rate(sorted_heroes[4],sorted_heroes[0])*1.1**4,0,0,get_counter_rate(sorted_heroes[4],sorted_heroes[3])*1.1**4,0,0,0,0,0,0],
                            [0,get_counter_rate(sorted_heroes[5],sorted_heroes[1])*1.1**5,get_counter_rate(sorted_heroes[5],sorted_heroes[2])*1.1**5,0,get_counter_rate(sorted_heroes[5],sorted_heroes[4])*1.1**5,0,0,0,0,0],
                            [0,get_counter_rate(sorted_heroes[6],sorted_heroes[1])*1.1**6,get_counter_rate(sorted_heroes[6],sorted_heroes[2])*1.1**6,0,get_counter_rate(sorted_heroes[6],sorted_heroes[4])*1.1**6,0,0,0,0,0],
                            [get_counter_rate(sorted_heroes[7],sorted_heroes[0])*1.1**7,0,0,get_counter_rate(sorted_heroes[7],sorted_heroes[3])*1.1**7,0,get_counter_rate(sorted_heroes[7],sorted_heroes[5])*1.1**7,get_counter_rate(sorted_heroes[7],sorted_heroes[6])*1.1**7,0,0,0],
                            [0,get_counter_rate(sorted_heroes[8],sorted_heroes[1])*1.1**8,get_counter_rate(sorted_heroes[8],sorted_heroes[2])*1.1**8,0,get_counter_rate(sorted_heroes[8],sorted_heroes[4])*1.1**8,0,0,get_counter_rate(sorted_heroes[8],sorted_heroes[7])*1.1**8,0,0],
                            [get_counter_rate(sorted_heroes[9],sorted_heroes[0])*1.1**9,0,0,get_counter_rate(sorted_heroes[9],sorted_heroes[3])*1.1**9,0,get_counter_rate(sorted_heroes[9],sorted_heroes[5])*1.1**9,get_counter_rate(sorted_heroes[9],sorted_heroes[6])*1.1**9,0,get_counter_rate(sorted_heroes[9],sorted_heroes[8])*1.1**9,0]])
    counter_matrix[i, :, :] = counter_rate   
    # print(sorted_heroes)
    picks = np.array([hero_vec[hero] for hero in sorted_heroes])
    X[i, :, :] = picks
    print(i)
    # get match result
    y[i] = int(row['radiant_win'])

np.savez('data.npz', X=X, y=y, counter_matrix=counter_matrix)

            

