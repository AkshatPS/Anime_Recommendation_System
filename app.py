import pickle
import streamlit as st
import requests
from bs4 import BeautifulSoup
from fastapi import FastAPI
from starlette.responses import RedirectResponse

app = FastAPI()

@app.get("/")
async def read_root():
    return {"Hello": "World"}

@app.get("/favicon.ico")
async def favicon():
    return RedirectResponse(url="/")

def get_anime_poster(anime_id):
    if not anime_id:
        print("Anime ID is None.")
        return None

    anime_url = f"https://myanimelist.net/anime/{anime_id}"
    try:
        response = requests.get(anime_url)
        response.raise_for_status()

        soup = BeautifulSoup(response.content, 'html.parser')
        poster_img = soup.find('img', {'class': 'lazyload'})

        if poster_img:
            poster_url = poster_img.get('data-src') or poster_img.get('src')
            if poster_url:
                return poster_url
            else:
                print("Poster URL not found in image attributes.")
                return None
        else:
            print("Poster image not found.")
            return None
    except requests.exceptions.RequestException as e:
        print(f"Failed to retrieve anime page. Error: {e}")
        return None

def recommend(anime):
    try:
        index = animes[animes['name'] == anime].index[0]
    except IndexError:
        print(f"Anime '{anime}' not found in the list.")
        return [], []

    distances = sorted(list(enumerate(similarity[index])), reverse=True, key=lambda x: x[1])
    
    recommended_anime_names = []
    recommended_anime_posters = []
    
    for i in distances[1:6]:
        anime_id = animes.iloc[i[0]]['anime_id']
        if anime_id:
            poster_url = get_anime_poster(anime_id)
            if poster_url:
                recommended_anime_posters.append(poster_url)
                recommended_anime_names.append(animes.iloc[i[0]]['name'])
            else:
                recommended_anime_posters.append(None)
                recommended_anime_names.append(animes.iloc[i[0]]['name'])
        else:
            print(f"Anime ID not found for index {i[0]}")

    return recommended_anime_names, recommended_anime_posters

st.header('Anime Recommender System')
animes = pickle.load(open('anime_list.pkl', 'rb'))
similarity = pickle.load(open('similarity.pkl', 'rb'))

anime_list = animes['name'].values
selected_anime = st.selectbox("Type or select an anime from the dropdown", anime_list)

if st.button('Show Recommendation'):
    recommended_anime_names, recommended_anime_posters = recommend(selected_anime)

    col1, col2, col3, col4, col5 = st.columns(5)
    columns = [col1, col2, col3, col4, col5]

    for idx, col in enumerate(columns):
        if idx < len(recommended_anime_names):
            with col:
                st.text(recommended_anime_names[idx])
                if recommended_anime_posters[idx]:
                    st.image(recommended_anime_posters[idx])  # Display the image in its original size
                else:
                    st.text("Poster not available")
