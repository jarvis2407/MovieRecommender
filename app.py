import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# =====================================
# PAGE CONFIG
# =====================================

st.set_page_config(
    page_title="Movie Recommendation System",
    page_icon="🎬",
    layout="wide"
)

# =====================================
# CUSTOM CSS
# =====================================

st.markdown("""
<style>

.main {
    background-color: #0E1117;
    color: white;
}

.title {
    text-align: center;
    font-size: 50px;
    font-weight: bold;
    color: #FF4B4B;
    margin-bottom: 10px;
}

.subtitle {
    text-align: center;
    font-size: 20px;
    color: #BBBBBB;
    margin-bottom: 40px;
}

.stButton > button {
    width: 100%;
    background-color: #FF4B4B;
    color: white;
    border-radius: 10px;
    height: 50px;
    font-size: 18px;
    border: none;
}

.stButton > button:hover {
    background-color: #ff1f1f;
    color: white;
}

.movie-card {
    background-color: #1E1E1E;
    padding: 20px;
    border-radius: 15px;
    text-align: center;
    margin-top: 20px;
    box-shadow: 0px 4px 10px rgba(0,0,0,0.3);
}

</style>
""", unsafe_allow_html=True)

# =====================================
# LOAD DATASET
# =====================================

movies = pd.read_csv("indian movies.csv")

# Keep useful columns
movies = movies[['Movie Name', 'Genre', 'Language', 'Year']]




# Remove null values
movies.dropna(inplace=True)
movies = movies.head(5000)


movies['tags'] = (
    movies['Genre'] + " " +
    movies['Language'] + " " +
    movies['Year'].astype(str)
)

# Convert to lowercase
movies['tags'] = movies['tags'].str.lower()

# =====================================
# VECTORIZATION
# =====================================

cv = CountVectorizer(max_features=5000, stop_words='english')

vectors = cv.fit_transform(movies['tags']).toarray()

# =====================================
# SIMILARITY MATRIX
# =====================================

similarity = cosine_similarity(vectors)

# =====================================
# RECOMMEND FUNCTION
# =====================================

def recommend(movie):

    recommended_movies = []

    movie_index = movies[movies['Movie Name'] == movie].index[0]

    distances = similarity[movie_index]

    movies_list = sorted(
        list(enumerate(distances)),
        reverse=True,
        key=lambda x: x[1]
    )[1:6]

    for i in movies_list:

        movie_name = movies.iloc[i[0]]['Movie Name']

        recommended_movies.append(movie_name)

    return recommended_movies

# =====================================
# UI
# =====================================

st.markdown(
    '<div class="title">🎬 Movie Recommendation System</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Find movies similar to your favorites </div>',
    unsafe_allow_html=True
)

# Movie dropdown
movie_list = movies['Movie Name'].values

selected_movie = st.selectbox(
    "Type or Select a Movie",
    movie_list
)

# Recommend button
if st.button("Recommend"):

    recommendations = recommend(selected_movie)

    st.subheader("Recommended Movies")

    col1, col2, col3, col4, col5 = st.columns(5)

    cols = [col1, col2, col3, col4, col5]

    for col, movie in zip(cols, recommendations):

        with col:
            st.markdown(
                f"""
                <div class="movie-card">
                    <h3>{movie}</h3>
                </div>
                """,
                unsafe_allow_html=True
            )