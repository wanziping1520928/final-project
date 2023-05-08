from gensim.models import Word2Vec
import pandas
import matplotlib.pyplot as plt
import seaborn as sns
import adjustText
from sklearn.decomposition import PCA
import requests

model = Word2Vec.load('word2vec.model1')
# mode2 = Word2Vec.load('word2vec.mode2')
# print(model.wv.most_similar('Luna'))

def plot_2d_representation_of_words(word_list,word_vectors,label_x_axis="x",label_y_axis="y",label_label="heros"):
    pca = PCA(n_components=2)
    word_plus_coordinates = []
    for word in word_list:
        current_row = []
        current_row.append(word)
        current_row.extend(word_vectors.wv[word])#get the word vector
        word_plus_coordinates.append(current_row)
    word_plus_coordinates = pandas.DataFrame(word_plus_coordinates)
    print(word_plus_coordinates)
    coordinates_2d = pca.fit_transform(word_plus_coordinates.iloc[:, 1:])
    coordinates_2d = pandas.DataFrame(coordinates_2d, columns=[label_x_axis, label_y_axis])
    coordinates_2d[label_label] = word_plus_coordinates.iloc[:, 0]
    plt.figure(figsize=(20, 15))
    p1 = sns.scatterplot(data=coordinates_2d, x=label_x_axis, y=label_y_axis)
    sns.set(font_scale = 1.3)
    x = coordinates_2d[label_x_axis]
    y = coordinates_2d[label_y_axis]
    label = coordinates_2d[label_label]
    texts = [plt.text(x[i], y[i], label[i]) for i in range(len(x))]
    adjustText.adjust_text(texts)
    plt.show()


response = requests.get('https://api.opendota.com/api/heroes')
data = response.json()
heroname = [hero['localized_name']for hero in data[:-1]]
print(len(heroname))


plot_2d_representation_of_words(word_list = heroname,word_vectors = model,label_x_axis="x",label_y_axis="y",label_label="heros")

# plot_2d_representation_of_words(word_list = heroname,word_vectors = mode2,label_x_axis="x",label_y_axis="y",label_label="heros")