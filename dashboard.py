import streamlit as st
import pandas as pd
import requests
import random
import plotly.express as px
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Netflix AI Dashboard", layout="wide")

# -------- NETFLIX STYLE --------
st.markdown("""
<style>
body {background-color:#0e1117;}
h1,h2,h3 {color:#E50914;}
</style>
""", unsafe_allow_html=True)

# -------- HERO --------
st.title("🎬 Netflix Analytics Dashboard")

st.markdown("""
### Welcome to Omprakash's Netflix Analytics Dashboard  
Explore movies, discover trends, and get AI-based recommendations.
""")

# -------- LOAD DATA --------
df = pd.read_csv("netflix_titles.csv")

# -------- TMDB API --------
API_KEY = "YOUR_TMDB_API_KEY"

def fetch_poster(movie):

    url=f"https://api.themoviedb.org/3/search/movie?api_key={API_KEY}&query={movie}"

    data=requests.get(url).json()

    try:
        poster=data["results"][0]["poster_path"]
        return "https://image.tmdb.org/t/p/w500"+poster
    except:
        return "https://via.placeholder.com/200x300"

# -------- SIDEBAR --------
st.sidebar.header("Filters")

type_filter = st.sidebar.selectbox(
"Content Type",
df["type"].dropna().unique()
)

year_filter = st.sidebar.slider(
"Release Year",
int(df["release_year"].min()),
int(df["release_year"].max()),
2018
)

filtered_df=df[
(df["type"]==type_filter)&
(df["release_year"]<=year_filter)
]

# -------- SEARCH --------
st.subheader("🔎 Search Movie")

search=st.text_input("Search movie name")

if search:

    result=filtered_df[
    filtered_df["title"].str.contains(search,case=False)
    ]

    st.dataframe(result[["title","type","country","release_year"]])

# -------- TRENDING SECTION --------
st.subheader("🔥 Trending Movies")

trending=df.sample(10)["title"].tolist()

cols=st.columns(5)

for i,movie in enumerate(trending):

    poster=fetch_poster(movie)

    cols[i%5].image(poster,caption=movie)

# -------- NETFLIX STYLE SCROLLING --------
st.subheader("🎥 Popular on Netflix")

movies=df.sample(20)["title"].tolist()

scroll_cols=st.columns(10)

for i,m in enumerate(movies):

    poster=fetch_poster(m)

    scroll_cols[i%10].image(poster,caption=m)

# -------- AI RECOMMENDATION --------
st.subheader("🤖 AI Movie Recommendation")

df["combined"]=df["title"]+df["director"].fillna("")

vectorizer=CountVectorizer(stop_words="english")

matrix=vectorizer.fit_transform(df["combined"])

similarity=cosine_similarity(matrix)

movie_list=df["title"].dropna().unique()

selected=st.selectbox("Choose Movie",movie_list)

if st.button("Recommend"):

    index=df[df["title"]==selected].index[0]

    scores=list(enumerate(similarity[index]))

    scores=sorted(scores,key=lambda x:x[1],reverse=True)[1:6]

    cols=st.columns(5)

    for i,s in enumerate(scores):

        movie=df.iloc[s[0]].title

        poster=fetch_poster(movie)

        cols[i].image(poster,caption=movie)

# -------- POSTER GALLERY --------
st.subheader("🎬 Movie Gallery")

sample=df.sample(40)["title"].tolist()

cols=st.columns(8)

for i,m in enumerate(sample):

    poster=fetch_poster(m)

    cols[i%8].image(poster,caption=m)

# -------- CHARTS (BOTTOM) --------
st.markdown("---")
st.header("📊 Netflix Analytics")

type_chart=px.pie(
df,
names="type",
title="Movies vs TV Shows"
)

st.plotly_chart(type_chart)

country_chart=px.bar(
df["country"].value_counts().head(10),
title="Top Countries"
)

st.plotly_chart(country_chart)

year_chart=px.histogram(
df,
x="release_year",
title="Release Year Distribution"
)

st.plotly_chart(year_chart)

# -------- FOOTER --------
st.markdown("---")

st.markdown("""
### 👨‍💻 Developed by Omprakash Dwivedi  
B.Tech CSE | Data Analytics & AI Enthusiast
""")
