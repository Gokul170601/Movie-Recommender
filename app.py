import streamlit as st 
import pandas as pd
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import imdb

#itle
st.title('Movie Recommender System')

#load the pickle model
model = pickle.load(open('model.pkl', 'rb'))
df = pd.read_csv('https://raw.githubusercontent.com/Gokul170601/Movie-Recommender/main/processed_data.csv')

#selecting movie
selected = st.selectbox('Select a Movie', options=df['movie_title'], index=None)

#function for fetch the movie details
def get_movie_details(imdb_id):
    ia = imdb.IMDb()
    movie = ia.get_movie(imdb_id[2:])
    details = {
        'title': movie.get('title'),
        'cover_url': movie.get('cover url'),
        'director': [director['name'] for director in movie.get('director', [])],
        'plot': movie.get('plot outline'),
        'year': movie.get('year')
    }
    return details

#function for recommend movie
def recommend(movie):
    idx = df[df['movie_title'] == movie].index[0]
    similarity = cosine_similarity(model[idx], model)
    sorted_list = sorted(list(enumerate(similarity[0])), reverse=True, key=lambda x: x[1])[1:6]

    recommend_details = []
    for i in sorted_list:
        movie_id = df.iloc[i[0]].movie_id
        details = get_movie_details(movie_id)
        recommend_details.append(details)
        
    return recommend_details

#display the recommended movies in app
if st.button('Recommend'):
    recommendations = recommend(movie=selected)

    for details in recommendations:
        col1, col2 = st.columns([1, 2])
        
        with col1:
            if details['cover_url']:
                st.image(details['cover_url'], width=150)
        
        with col2:
            st.subheader(details['title'])
            st.write(f"Director: {', '.join(details['director'])}")
            st.write(f"Year: {details['year']}")
            st.write(f"Plot: {details['plot']}")


