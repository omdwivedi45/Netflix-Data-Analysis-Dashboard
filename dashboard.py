import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.title("Netflix Data Analysis Dashboard")

data = pd.read_csv("netflix_titles.csv")

st.subheader("Dataset Preview")

st.dataframe(data.head())

# Movies vs TV Shows

st.subheader("Movies vs TV Shows")

type_count = data['type'].value_counts()

fig, ax = plt.subplots()

type_count.plot(kind='bar', ax=ax)

st.pyplot(fig)

# Top Countries

st.subheader("Top Countries")

country_count = data['country'].value_counts().head(10)

fig2, ax2 = plt.subplots()

country_count.plot(kind='bar', ax=ax2)

st.pyplot(fig2)

# Genre analysis

st.subheader("Top Genres")

genre = data['listed_in'].str.split(',').explode()

genre_count = genre.value_counts().head(10)

fig3, ax3 = plt.subplots()

genre_count.plot(kind='bar', ax=ax3)

st.pyplot(fig3)

# Year filter

st.subheader("Filter by Release Year")

year = st.slider("Select Year", 2000, 2021, 2018)

filtered = data[data['release_year'] == year]

st.write(filtered[['title','type','country']])