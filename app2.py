import pickle
import streamlit as st
import requests
from fastapi import FastAPI
from starlette.responses import RedirectResponse

app = FastAPI()

@app.get("/")
async def read_root():
    return {"Hello": "World"}

@app.get("/favicon.ico")
async def favicon():
    return RedirectResponse(url="/")

# AniList API function for getting the poster
def get_anime_poster(anime_name):
    query = '''
    query ($search: String) {
      Media(search: $search, type: ANIME) {
        coverImage {
          large
        }
      }
    }
    '''
    variables = {
        'search': anime_name
    }
    url = 'https://graphql.anilist.co'
    try:
        response = requests.post(url, json={'query': query, 'variables': variables})
        response.raise_for_status()
        data = response.json()
        poster_url = data['data']['Media']['coverImage']['large']
        return poster_url
    except requests.exceptions.RequestException as e:
        print(f"Error retrieving data from AniList API: {e}")
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
        anime_name = animes.iloc[i[0]]['name']
        if anime_name:
            poster_url = get_anime_poster(anime_name)  # Fetch the poster using AniList API
            if poster_url:
                recommended_anime_posters.append(poster_url)
                recommended_anime_names.append(anime_name)
            else:
                recommended_anime_posters.append(None)
                recommended_anime_names.append(anime_name)
        else:
            print(f"Anime name not found for index {i[0]}")

    return recommended_anime_names, recommended_anime_posters


st.header('Anime Recommender System')


st.markdown("""
    <style>
    .anime-title {
        white-space: normal;  /* Allows the text to wrap to the next line */
        overflow: visible;    /* Makes sure there's no hidden overflow */
        text-align: center;   /* Centers the text */
        font-size: 14px;      /* Adjusts the font size */
        font-weight: bold;    /* Makes the text bold */
        margin-top: 10px;     /* Adds some margin from the top */
        padding: 5px;         /* Padding around the text */
        word-wrap: break-word; /* Ensures long words wrap correctly */
    }
    .stImage {
        margin-left: auto;
        margin-right: auto;
        display: block;
        width: 100%;
        height: auto;
    }
    </style>
    """, unsafe_allow_html=True)

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
                # Display anime names with proper text wrapping
                st.markdown(f"<div class='anime-title'>{recommended_anime_names[idx]}</div>", unsafe_allow_html=True)
                if recommended_anime_posters[idx]:
                    st.image(recommended_anime_posters[idx], use_column_width=True)  # Automatically adjusts image size
                else:
                    st.text("Poster not available")

