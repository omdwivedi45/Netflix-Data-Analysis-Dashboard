import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Page configuration
st.set_page_config(
    page_title="Netflix Data Analysis Dashboard",
    page_icon="🎬",
    layout="wide"
)

st.title("🎬 Netflix Data Analysis Dashboard")

# Load dataset
data = pd.read_csv("netflix_titles.csv")

# ---------------- SIDEBAR FILTERS ----------------

st.sidebar.header("Filter Movies")

type_filter = st.sidebar.selectbox(
    "Select Type",
    data["type"].dropna().unique()
)

year_filter = st.sidebar.slider(
    "Select Release Year",
    int(data["release_year"].min()),
    int(data["release_year"].max()),
    int(data["release_year"].max())
)

filtered_df = data[
    (data["type"] == type_filter) &
    (data["release_year"] <= year_filter)
]

# ---------------- DATA PREVIEW ----------------

st.subheader("Dataset Preview")
st.dataframe(filtered_df.head())

# ---------------- MOVIES VS TV SHOWS ----------------

st.subheader("Movies vs TV Shows")

type_count = filtered_df['type'].value_counts()

fig, ax = plt.subplots()
type_count.plot(kind='bar', ax=ax)
st.pyplot(fig)

# ---------------- TOP COUNTRIES ----------------

st.subheader("Top Countries")

country_count = filtered_df['country'].value_counts().head(10)

fig2, ax2 = plt.subplots()
country_count.plot(kind='bar', ax=ax2)
st.pyplot(fig2)

# ---------------- GENRE ANALYSIS ----------------

st.subheader("Top Genres")

genre = filtered_df['listed_in'].str.split(',').explode()
genre_count = genre.value_counts().head(10)

fig3, ax3 = plt.subplots()
genre_count.plot(kind='bar', ax=ax3)
st.pyplot(fig3)

# ---------------- YEAR FILTER TABLE ----------------

st.subheader("Movies Released in Selected Year")

year_movies = filtered_df[['title','type','country','release_year']]
st.write(year_movies)

# ---------------- MOVIE POSTERS SECTION ----------------

st.subheader("Featured Movies")

poster_urls = [
"https://image.tmdb.org/t/p/w500/8UlWHLMpgZm9bx6QYh0NFoq67TZ.jpg",
"https://image.tmdb.org/t/p/w500/q6y0Go1tsGEsmtFryDOJo3dEmqu.jpg",
"https://image.tmdb.org/t/p/w500/5KCVkau1HEl7ZzfPsKAPM0sMiKc.jpg"
]

cols = st.columns(3)

for col, poster in zip(cols, poster_urls):
    col.image(poster)
