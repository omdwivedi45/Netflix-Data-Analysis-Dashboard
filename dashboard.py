import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Page config
st.set_page_config(page_title="Netflix AI Dashboard", layout="wide")

st.title("🎬 Netflix Movies Dashboard")

# Load dataset
df = pd.read_csv("netflix_titles.csv")

# ---------------- SIDEBAR ----------------

st.sidebar.header("Filters")

type_filter = st.sidebar.selectbox(
    "Select Type",
    df["type"].dropna().unique()
)

year_filter = st.sidebar.slider(
    "Release Year",
    int(df["release_year"].min()),
    int(df["release_year"].max()),
    int(df["release_year"].max())
)

filtered_df = df[
    (df["type"] == type_filter) &
    (df["release_year"] <= year_filter)
]

# ---------------- DATA PREVIEW ----------------

st.subheader("Dataset Preview")
st.dataframe(filtered_df.head())

# ---------------- ANALYTICS ----------------

st.subheader("Movies vs TV Shows")
st.bar_chart(filtered_df["type"].value_counts())

st.subheader("Top Countries")
st.bar_chart(filtered_df["country"].value_counts().head(10))

st.subheader("Top Genres")

genres = filtered_df["listed_in"].str.split(",").explode()
st.bar_chart(genres.value_counts().head(10))

# ---------------- SEARCH ----------------

st.subheader("🔎 Search Movie")

search = st.text_input("Enter movie name")

if search:
    results = filtered_df[
        filtered_df["title"].str.contains(search, case=False)
    ]
    st.write(results[["title","type","country","release_year"]])

# ---------------- RECOMMENDATION ----------------

st.subheader("🤖 Movie Recommendation")

df["combined"] = df["title"] + df["director"].fillna("")

vectorizer = CountVectorizer(stop_words="english")
matrix = vectorizer.fit_transform(df["combined"])

similarity = cosine_similarity(matrix)

movie_list = df["title"].dropna().unique()

selected_movie = st.selectbox("Choose Movie", movie_list)

if st.button("Recommend"):

    index = df[df["title"] == selected_movie].index[0]

    scores = list(enumerate(similarity[index]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:6]

    st.write("Recommended Movies:")

    for i in scores:
        st.write(df.iloc[i[0]].title)

# ---------------- MOVIE POSTER GALLERY ----------------

st.subheader("🎬 Popular Movies Gallery")

poster_urls = [
"https://image.tmdb.org/t/p/w500/9O1Iy9od7b0T1gM6h0Zq9KoU3Lj.jpg",
"https://image.tmdb.org/t/p/w500/5P8SmMzSNYikXpxil6BYzJ16611.jpg",
"https://image.tmdb.org/t/p/w500/udDclJoHjfjb8Ekgsd4FDteOkCU.jpg",
"https://image.tmdb.org/t/p/w500/t6HIqrRAclMCA60NsSmeqe9RmNV.jpg",
"https://image.tmdb.org/t/p/w500/3bhkrj58Vtu7enYsRolD1fZdja1.jpg",
"https://image.tmdb.org/t/p/w500/q719jXXEzOoYaps6babgKnONONX.jpg",
"https://image.tmdb.org/t/p/w500/kqjL17yufvn9OVLyXYpvtyrFfak.jpg",
"https://image.tmdb.org/t/p/w500/yXNVcG0C7Oymg9F9ecXa9MWVwj8.jpg",
"https://image.tmdb.org/t/p/w500/8UlWHLMpgZm9bx6QYh0NFoq67TZ.jpg",
"https://image.tmdb.org/t/p/w500/5KCVkau1HEl7ZzfPsKAPM0sMiKc.jpg"
]

movie_names = [
"Sholay","3 Idiots","Dangal","PK","Lagaan",
"Gangs of Wasseypur","Zindagi Na Milegi Dobara",
"Queen","Bajrangi Bhaijaan","Andhadhun"
]

poster_urls = poster_urls * 5
movie_names = movie_names * 5

cols = st.columns(5)

for i, (poster, name) in enumerate(zip(poster_urls, movie_names)):
    cols[i % 5].image(poster, caption=name)
