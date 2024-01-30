import numpy as np
import pandas as pd
movies = pd.read_csv('/content/drive/MyDrive/DataSheet/tmdb_5000_movies.csv')
credit = pd.read_csv('/content/drive/MyDrive/DataSheet/tmdb_5000_credits.csv')
movies = movies.merge(credit,on='title')
movies.shape
movies.head(1)
#genres
#id
#keyword
#cast
#crew
movies.info()
#movies['original_language'].value_counts()
movies=movies[['id','title','overview','genres','keywords','cast','crew']]
print(movies)
movies.isnull().sum()
movies.dropna(inplace=True)   #deleting null data
movies.duplicated().sum()       #checking duplicate value
movies.iloc[0].genres
#[{"id": 28, "name": "Action"}, {"id": 12, "name": "Adventure"}, {"id": 14, "name": "Fantasy"}, {"id": 878, "name": "Science Fiction"}]
#['Action','Adventure','Fantasy','Science Fiction']
#since above list are not actual list it's a string
import ast
def convert(obj):
  l = []
  for i in ast.literal_eval(obj):
    l.append(i['name'])
  return l
movies['genres']=movies['genres'].apply(convert)
print(movies.head(1))
movies['genres']=movies['genres'].apply(convert)
print(movies.head(1))
movies['cast'][0]   #we need name of only 3 actress
def convert3(obj):
  l = []
  counter = 0
  for i in ast.literal_eval(obj):
    if counter != 3:
      l.append(i['name'])
      counter+=1
    else:
      break
  return l
movies['cast'] = movies['cast'].apply(convert3)
print(movies.head())
movies['crew'][0]   #we need only the name of director
def director(obj):
  l = []
  for i in ast.literal_eval(obj):
    if i['job']=='Director':
      l.append(i['name'])
      break
  return l
def director(obj):
  l = []
  for i in ast.literal_eval(obj):
    if i['job']=='Director':
      l.append(i['name'])
      break
  return l
print(movies.head(1))
movies['overview'] = movies['overview'].apply(lambda x: x.split())
from os import replace
movies['genres']=movies['genres'].apply(lambda x:[i.replace(" ","") for i in x])
movies['keywords'] = movies['keywords'].apply(lambda x:[i.replace(" ","")for i in x])
movies['cast'] = movies['cast'].apply(lambda x:[i.replace(" ","")for i in x])
movies['crew'] = movies['crew'].apply(lambda x:[i.replace(" ","")for i in x])
print(movies.head())
movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']
movies['tags']
new_df = movies[['id','title','tags']]
print(new_df)
new_df = movies[['id','title','tags']]
print(new_df)
new_df['tags']=new_df['tags'].apply(lambda x:" ".join(x))
import nltk
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
def stem(text):
  y = []
  for i in text.split():
    y.append(ps.stem(i))
  return " ".join(y)
new_df['tags'] = new_df['tags'].apply(stem)
new_df['tags'][0]
new_df['tags'] = new_df['tags'].apply(lambda x: x.lower())
new_df['tags'][0]
new_df['tags'][1]
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=5000,stop_words='english')
vectors = cv.fit_transform(new_df['tags']).toarray()
print(vectors[0])
cv.get_feature_names_out()
from sklearn.metrics.pairwise import cosine_similarity
similarity =cosine_similarity(vectors)
print(similarity.shape)
print(similarity)
print(similarity[0])
print(similarity[0].shape)
print(new_df)
new_df[new_df['title'] =='Avatar'].index[0]
new_df[new_df['title'] =='Batman Begins'].index[0]
def recommend(movie):
  movie_index = new_df[new_df['title'] ==movie].index[0]
  distances = similarity[movie_index]
  movie_list = sorted(list(enumerate(distances)),reverse = True,key=lambda x:x[1])[1:6]
  for i in movie_list:
    print(new_df.iloc[i[0]].title)

recommend('Superman')