import streamlit as st
import pandas as pd
import random
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Netflix AI Dashboard", layout="wide")

# ---------- TITLE ----------
st.title("🎬 Netflix Movies Dashboard")

# ---------- HERO ----------
st.markdown("""
### Welcome to Omprakash's Netflix Analytics Dashboard
Explore movies, discover trends, and get AI-based recommendations.
""")

# ---------- LOAD DATA ----------
df = pd.read_csv("netflix_titles.csv")

# ---------- SIDEBAR ----------
st.sidebar.header("Filters")

type_filter = st.sidebar.selectbox(
    "Content Type",
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

# ---------- METRICS ----------
col1, col2, col3 = st.columns(3)

col1.metric("Total Titles", len(df))
col2.metric("Movies", len(df[df["type"]=="Movie"]))
col3.metric("TV Shows", len(df[df["type"]=="TV Show"]))

# ---------- ANALYTICS ----------
st.subheader("📊 Content Distribution")
st.bar_chart(filtered_df["type"].value_counts())

st.subheader("🌍 Top Countries")
st.bar_chart(filtered_df["country"].value_counts().head(10))

st.subheader("🎭 Top Genres")

genres = filtered_df["listed_in"].str.split(",").explode()
st.bar_chart(genres.value_counts().head(10))

# ---------- SEARCH ----------
st.subheader("🔎 Search Movie")

search = st.text_input("Enter movie name")

if search:
    results = filtered_df[
        filtered_df["title"].str.contains(search, case=False)
    ]
    st.write(results[["title","type","country","release_year"]])

# ---------- RECOMMENDATION ----------
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

# ---------- TRENDING ----------
st.subheader("🔥 Trending Movies")

trending = [
"3 Idiots","Dangal","PK","Andhadhun","Queen",
"Bajrangi Bhaijaan","Sholay","Lagaan","Drishyam","Barfi"
]

for movie in trending:
    st.write("⭐", movie)

# ---------- MOVIE POSTERS ----------
st.subheader("🎬 Movie Poster Gallery")

# sample 500 titles randomly
sample_movies = df.sample(min(500, len(df)))

# fake poster generator (random placeholder)
def get_poster():
    return f"https://picsum.photos/200/300?random={random.randint(1,10000)}"

posters = [get_poster() for _ in range(len(sample_movies))]

movies = sample_movies["title"].tolist()

# pagination
page = st.slider("Select Page", 1, int(len(posters)/20)+1, 1)

start = (page-1)*20
end = start+20

cols = st.columns(5)

for i, (poster, name) in enumerate(zip(posters[start:end], movies[start:end])):
    cols[i % 5].image(poster, caption=name)

# ---------- FOOTER ----------
st.markdown("---")

st.markdown("""
Developed by **Omprakash Dwivedi**  
B.Tech CSE | Data Analytics & AI Enthusiast
""")
