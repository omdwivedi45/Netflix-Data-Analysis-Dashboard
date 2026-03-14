import streamlit as st
import pandas as pd
import requests
import plotly.express as px
import random
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Netflix AI Dashboard", layout="wide")

# -------- NETFLIX STYLE UI --------
st.markdown("""
<style>
body {background-color:#0e1117;}
h1,h2,h3 {color:#E50914;}
.stButton>button {background:#E50914;color:white;}
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

# -------- WATCHLIST --------
if "watchlist" not in st.session_state:
    st.session_state.watchlist = []

# -------- TMDB POSTER --------
API_KEY = "YOUR_TMDB_API_KEY"

def get_poster(title):

    try:

        url=f"https://api.themoviedb.org/3/search/movie?api_key={API_KEY}&query={title}"
        data=requests.get(url).json()

        poster=data["results"][0]["poster_path"]

        return "https://image.tmdb.org/t/p/w500"+poster

    except:

        # fallback image
        return f"https://picsum.photos/200/300?random={random.randint(1,10000)}"

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

search=st.text_input("Search movie")

if search:

    result=filtered_df[
    filtered_df["title"].str.contains(search,case=False)
    ]

    st.dataframe(result[["title","type","release_year"]])

# -------- TRENDING CAROUSEL --------
st.subheader("🔥 Trending Now")

movies=df.sample(15)["title"].tolist()

cols=st.columns(5)

for i,m in enumerate(movies):

    poster=get_poster(m)

    with cols[i%5]:

        st.image(poster)

        st.caption(m)

        if st.button(f"❤️ Add",key=m):

            st.session_state.watchlist.append(m)

# -------- WATCHLIST --------
st.subheader("❤️ Your Watchlist")

if len(st.session_state.watchlist)==0:

    st.write("No movies added yet")

else:

    watch_cols=st.columns(5)

    for i,m in enumerate(st.session_state.watchlist):

        poster=get_poster(m)

        watch_cols[i%5].image(poster,caption=m)

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

    rec_cols=st.columns(5)

    for i,s in enumerate(scores):

        movie=df.iloc[s[0]].title

        poster=get_poster(movie)

        rec_cols[i].image(poster,caption=movie)

# -------- MOVIE GALLERY --------
st.subheader("🎬 Popular Movies")

sample=df.sample(30)["title"].tolist()

cols=st.columns(6)

for i,m in enumerate(sample):

    poster=get_poster(m)

    cols[i%6].image(poster,caption=m)

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
