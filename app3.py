'''
Author: Bappy Ahmed
Email: entbappy73@gmail.com
Date: 2021-Nov-15
'''

import pickle
import streamlit as st
import requests

def fetch_poster(movie_id):
    api_key = "YOUR_API_KEY"  # Replace with your actual API key
    base_url = "https://image.tmdb.org/t/p/w500"

    try:
        url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={api_key}"
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()

        print(f"Movie ID: {movie_id} -> API Response: {data}")  # Debugging

        poster_path = data.get('poster_path')
        if poster_path:
            return base_url + poster_path
        else:
            print(f"❌ No poster found for movie ID {movie_id}")
            return "https://via.placeholder.com/300x450.png?text=No+Image"

    except requests.exceptions.RequestException as e:
        print(f"🚨 Error fetching poster for movie ID {movie_id}: {e}")
        return "https://via.placeholder.com/300x450.png?text=No+Image"



def recommend(movie):
    index = movies[movies['title'] == movie].index[0]
    distances = sorted(list(enumerate(similarity[index])), reverse=True, key=lambda x: x[1])
    recommended_movie_names = []
    recommended_movie_posters = []
    for i in distances[1:6]:
        # fetch the movie poster
        movie_id = movies.iloc[i[0]].movie_id
        recommended_movie_posters.append(fetch_poster(movie_id))
        recommended_movie_names.append(movies.iloc[i[0]].title)

    return recommended_movie_names,recommended_movie_posters


st.header('Movie Recommender')
movies = pickle.load(open('artifacts/movie_list.pkl','rb'))
similarity = pickle.load(open('artifacts/similarity.pkl','rb'))

movie_list = movies['title'].values
selected_movie = st.selectbox(
    "Type or select a movie from the dropdown",
    movie_list
)

if st.button('Show Recommendation'):
    recommended_movie_names,recommended_movie_posters = recommend(selected_movie)
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.text(recommended_movie_names[0])
    with col2:
        st.text(recommended_movie_names[1])

    with col3:
        st.text(recommended_movie_names[2])

    with col4:
        st.text(recommended_movie_names[3])

    with col5:
        st.text(recommended_movie_names[4])


