import streamlit as st
import pandas as pd
import requests
import plotly.express as px
import random
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Netflix AI Dashboard", layout="wide")

# ----------- CSS -----------
st.markdown("""
<style>
body {background-color:#0e1117;}
h1,h2,h3 {color:#E50914;}
.stButton>button {background:#E50914;color:white;}
.hero {
background-image:url("https://images.unsplash.com/photo-1489599849927-2ee91cede3ba");
background-size:cover;
padding:120px;
border-radius:10px;
text-align:center;
color:white;
}
.hero h1 {font-size:60px;}
.hero p {font-size:22px;}
</style>
""", unsafe_allow_html=True)

# ----------- LOGIN -----------
if "login" not in st.session_state:
    st.session_state.login=False

if not st.session_state.login:

    st.title("Login to Netflix Dashboard")

    user=st.text_input("Username")
    pwd=st.text_input("Password",type="password")

    if st.button("Login"):

        if user and pwd:
            st.session_state.login=True
            st.rerun()
        else:
            st.warning("Enter credentials")

    st.stop()

# ----------- HERO -----------
st.markdown("""
<div class="hero">
<h1>Unlimited Movies & Data Insights</h1>
<p>Netflix Analytics Dashboard with AI Recommendations</p>
</div>
""", unsafe_allow_html=True)

st.title("Netflix Analytics Dashboard")
st.write("Developed by Omprakash Dwivedi")

# ----------- DATA -----------
try:
    df=pd.read_csv("netflix_titles.csv")
except:
    st.error("Dataset missing")
    st.stop()

# ----------- APIs -----------
TMDB_API_KEY="YOUR_TMDB_KEY"
OMDB_API_KEY="YOUR_OMDB_KEY"

# ----------- WATCHLIST -----------
if "watchlist" not in st.session_state:
    st.session_state.watchlist=[]

# ----------- POSTER FUNCTION -----------
def get_poster(title):

    try:
        url=f"https://api.themoviedb.org/3/search/movie?api_key={TMDB_API_KEY}&query={title}"
        data=requests.get(url).json()

        poster=data["results"][0]["poster_path"]

        return "https://image.tmdb.org/t/p/w500"+poster
    except:
        return f"https://picsum.photos/200/300?random={random.randint(1,9999)}"

# ----------- IMDB RATING -----------
def get_rating(title):

    try:
        url=f"http://www.omdbapi.com/?t={title}&apikey={OMDB_API_KEY}"
        data=requests.get(url).json()

        return data["imdbRating"]
    except:
        return "N/A"

# ----------- SIDEBAR FILTERS -----------
st.sidebar.header("Filters")

type_filter=st.sidebar.selectbox(
"Content Type",
df["type"].dropna().unique()
)

year_filter=st.sidebar.slider(
"Release Year",
int(df["release_year"].min()),
int(df["release_year"].max()),
2018
)

filtered=df[
(df["type"]==type_filter)&
(df["release_year"]<=year_filter)
]

# ----------- SEARCH -----------
st.subheader("Search Movie")

search=st.text_input("Movie name")

if search:

    result=filtered[
    filtered["title"].str.contains(search,case=False)
    ]

    st.dataframe(result[["title","type","release_year"]])

# ----------- TRENDING ROW -----------
st.subheader("Trending Now")

movies=df.sample(10)["title"].tolist()

cols=st.columns(5)

for i,m in enumerate(movies):

    poster=get_poster(m)
    rating=get_rating(m)

    with cols[i%5]:

        st.image(poster)
        st.caption(f"{m} ⭐ {rating}")

        if st.button("Add Watchlist",key=m):
            st.session_state.watchlist.append(m)

# ----------- WATCHLIST -----------
st.subheader("Your Watchlist")

if len(st.session_state.watchlist)==0:

    st.write("No movies yet")

else:

    cols=st.columns(5)

    for i,m in enumerate(st.session_state.watchlist):

        poster=get_poster(m)
        rating=get_rating(m)

        cols[i%5].image(poster,caption=f"{m} ⭐ {rating}")

# ----------- AI RECOMMENDATION -----------
st.subheader("AI Recommendation")

df["combined"]=df["title"]+df["director"].fillna("")

vectorizer=CountVectorizer(stop_words="english")

matrix=vectorizer.fit_transform(df["combined"])

similarity=cosine_similarity(matrix)

movie_list=df["title"].dropna().unique()

selected=st.selectbox("Choose Movie",movie_list)

if st.button("Recommend Movies"):

    index=df[df["title"]==selected].index[0]

    scores=list(enumerate(similarity[index]))

    scores=sorted(scores,key=lambda x:x[1],reverse=True)[1:6]

    cols=st.columns(5)

    for i,s in enumerate(scores):

        movie=df.iloc[s[0]].title
        poster=get_poster(movie)
        rating=get_rating(movie)

        trailer=f"https://www.youtube.com/results?search_query={movie}+trailer"

        cols[i].image(poster,caption=f"{movie} ⭐ {rating}")
        cols[i].link_button("Trailer",trailer)

# ----------- MOVIE ROW -----------
st.subheader("Popular on Netflix")

sample=df.sample(24)["title"].tolist()

cols=st.columns(6)

for i,m in enumerate(sample):

    poster=get_poster(m)

    cols[i%6].image(poster,caption=m)

# ----------- CHARTS -----------
st.markdown("---")
st.header("Netflix Data Insights")

pie=px.pie(df,names="type",title="Movies vs TV Shows")
st.plotly_chart(pie)

country=px.bar(df["country"].value_counts().head(10),title="Top Countries")
st.plotly_chart(country)

year=px.histogram(df,x="release_year",title="Release Year Distribution")
st.plotly_chart(year)

# ----------- FOOTER -----------
st.markdown("---")

st.markdown("""
Developed by **Omprakash Dwivedi**  
B.Tech CSE | Data Analytics & AI Enthusiast
""")
