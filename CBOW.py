import pandas as pd
import requests
import numpy as np
from gensim.models import Word2Vec



response = requests.get('https://api.opendota.com/api/heroes')
data = response.json()
heroname = [hero['localized_name']for hero in data[:-1]]
print(len(heroname))

match_data = pd.read_csv('match_data.csv')
#去除无效数据
match_data = match_data.drop_duplicates('match_id')
match_data = match_data.dropna()
# print(match_data)

corpus = []
def get_train_data(match_data): #获取训练数据
    for index,row in match_data.iterrows():
        radiant_picks = eval(row['radiant_pick'])
        dire_picks = eval(row['dire_pick'])
        # print(radiant_picks)
        for i in range(5):
            corpus.append([radiant_picks[i]]+radiant_picks[:i]+radiant_picks[i+1:])
            corpus.append([dire_picks[i]]+dire_picks[:i]+dire_picks[i+1:])
    return corpus

get_train_data(match_data)
print(corpus)

model = Word2Vec(corpus, vector_size=150, window=5, min_count=1, workers=8, sg=0, epochs=10)
model.save('word2vec.model')

# model = Word2Vec(corpus, vector_size=150, window=5, min_count=1, workers=8, sg=1, epochs=10)
# model.save('word2vec.mode2')



    
