import streamlit as st
import pandas as pd
import requests
import plotly.express as px
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Netflix AI Dashboard", page_icon="🎬", layout="wide")

# =========================
# CONFIG
# =========================
TMDB_API_KEY = "YOUR_TMDB_KEY"   # <-- apni TMDB key yahan daalo

# =========================
# CSS
# =========================
st.markdown("""
<style>
.stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
    background-color: #141414 !important;
    color: white !important;
}

section[data-testid="stSidebar"] {
    background-color: #0b0b0b !important;
}

html, body, p, div, span, label {
    color: white !important;
}

h1, h2, h3, h4, h5 {
    color: white !important;
}

.netflix-red {
    color: #E50914 !important;
}

.hero-wrapper {
    position: relative;
    min-height: 100vh;
    border-radius: 18px;
    overflow: hidden;
    background:
        linear-gradient(to top, rgba(20,20,20,0.98) 5%, rgba(20,20,20,0.45) 35%, rgba(20,20,20,0.75) 100%),
        url('https://images.unsplash.com/photo-1489599849927-2ee91cede3ba?auto=format&fit=crop&w=1600&q=80');
    background-size: cover;
    background-position: center;
    padding: 24px 28px;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
}

.topbar {
    display: flex;
    justify-content: space-between;
    align-items: center;
}

.logo {
    font-size: 54px;
    font-weight: 900;
    color: #E50914 !important;
    letter-spacing: 1px;
}

.top-actions {
    display: flex;
    gap: 12px;
    align-items: center;
}

.lang-btn, .sign-btn, .start-btn {
    border: none;
    border-radius: 6px;
    padding: 10px 18px;
    font-weight: 700;
    cursor: pointer;
}

.lang-btn {
    background: rgba(0,0,0,0.6);
    color: white;
    border: 1px solid rgba(255,255,255,0.35);
}

.sign-btn, .start-btn {
    background: #E50914;
    color: white;
}

.hero-center {
    text-align: center;
    max-width: 900px;
    margin: 0 auto;
    padding-bottom: 80px;
}

.hero-center h1 {
    font-size: 64px;
    line-height: 1.05;
    font-weight: 900;
    margin-bottom: 18px;
}

.hero-center h3 {
    font-size: 28px;
    font-weight: 500;
    margin-bottom: 16px;
}

.hero-center p {
    font-size: 22px;
    margin-bottom: 24px;
}

.email-row {
    display: flex;
    justify-content: center;
    gap: 10px;
    flex-wrap: wrap;
}

.fake-input {
    min-width: 320px;
    max-width: 460px;
    width: 100%;
    background: rgba(20,20,20,0.85);
    border: 1px solid rgba(255,255,255,0.3);
    border-radius: 6px;
    padding: 16px 14px;
    color: #cfcfcf;
    font-size: 20px;
}

.section-box {
    background: #181818;
    border-radius: 16px;
    padding: 18px;
    margin-top: 18px;
}

.card-title {
    font-size: 28px;
    font-weight: 800;
    color: #E50914 !important;
    margin-bottom: 6px;
}

.stButton > button {
    background-color: #E50914 !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 700 !important;
}

div[data-baseweb="select"] > div,
input, textarea {
    background: #1f1f1f !important;
    color: white !important;
}

.poster-caption {
    text-align: center;
    font-size: 14px;
    margin-top: 6px;
}

@media (max-width: 768px) {
    .logo {
        font-size: 38px;
    }
    .hero-center h1 {
        font-size: 38px;
    }
    .hero-center h3 {
        font-size: 20px;
    }
    .hero-center p {
        font-size: 18px;
    }
}
</style>
""", unsafe_allow_html=True)

# =========================
# SESSION STATE
# =========================
if "page_mode" not in st.session_state:
    st.session_state.page_mode = "landing"

if "watchlist" not in st.session_state:
    st.session_state.watchlist = []

# =========================
# DATA LOAD
# =========================
try:
    df = pd.read_csv("netflix_titles.csv")
except Exception:
    st.error("netflix_titles.csv same folder mein rakho.")
    st.stop()

df = df.copy()
df["title"] = df["title"].fillna("Unknown")
df["director"] = df["director"].fillna("")
df["cast"] = df["cast"].fillna("Not available")
df["country"] = df["country"].fillna("Unknown")
df["listed_in"] = df["listed_in"].fillna("Unknown")
df["type"] = df["type"].fillna("Unknown")
df["release_year"] = pd.to_numeric(df["release_year"], errors="coerce").fillna(0).astype(int)

# =========================
# TMDB HELPERS
# =========================
def tmdb_search_movie(title: str):
    try:
        url = "https://api.themoviedb.org/3/search/movie"
        params = {"api_key": TMDB_API_KEY, "query": title}
        res = requests.get(url, params=params, timeout=10)
        data = res.json()
        if data.get("results"):
            return data["results"][0]
    except Exception:
        return None
    return None

def get_poster(title: str):
    movie = tmdb_search_movie(title)
    if movie and movie.get("poster_path"):
        return "https://image.tmdb.org/t/p/w500" + movie["poster_path"]
    return None

def get_backdrop(title: str):
    movie = tmdb_search_movie(title)
    if movie and movie.get("backdrop_path"):
        return "https://image.tmdb.org/t/p/original" + movie["backdrop_path"]
    return None

def get_tmdb_release_date(title: str):
    movie = tmdb_search_movie(title)
    if movie and movie.get("release_date"):
        return movie["release_date"]
    return None

# =========================
# LANDING PAGE
# =========================
def show_landing_page():
    st.markdown("""
    <div class="hero-wrapper">
        <div class="topbar">
            <div class="logo">NETFLIX</div>
            <div class="top-actions">
                <button class="lang-btn">English ▾</button>
                <button class="sign-btn">Sign In</button>
            </div>
        </div>

        <div class="hero-center">
            <h1>Unlimited movies, shows,<br>and more</h1>
            <h3>Starts at ₹149. Cancel at any time.</h3>
            <p>Ready to watch? Enter your email to create or restart your membership.</p>
            <div class="email-row">
                <div class="fake-input">Email address</div>
                <button class="start-btn">Get Started ❯</button>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.write("")
    c1, c2, c3 = st.columns([1, 1, 1])
    with c2:
        if st.button("Open Dashboard"):
            st.session_state.page_mode = "dashboard"
            st.rerun()

# =========================
# DASHBOARD PAGE
# =========================
def show_dashboard():
    top_left, top_right = st.columns([4, 1])
    with top_left:
        st.markdown("<h1 class='netflix-red'>Netflix AI Dashboard</h1>", unsafe_allow_html=True)
        st.write("Search movie, see poster, release date, cast and get recommendations.")
    with top_right:
        if st.button("Back to Home"):
            st.session_state.page_mode = "landing"
            st.rerun()

    st.sidebar.header("Filters")
    type_options = sorted(df["type"].dropna().unique().tolist())
    selected_type = st.sidebar.selectbox("Content Type", type_options)

    min_year = int(df["release_year"].min())
    max_year = int(df["release_year"].max())
    selected_year = st.sidebar.slider("Release Year", min_year, max_year, max_year)

    filtered_df = df[
        (df["type"] == selected_type) &
        (df["release_year"] <= selected_year)
    ]

    # SEARCH SECTION
    st.markdown("<div class='section-box'>", unsafe_allow_html=True)
    st.markdown("<div class='card-title'>🔎 Search Movie</div>", unsafe_allow_html=True)

    search_movie = st.text_input("Type a movie name")

    if search_movie:
        matches = filtered_df[
            filtered_df["title"].str.contains(search_movie, case=False, na=False)
        ]

        if not matches.empty:
            first_match = matches.iloc[0]
            movie_title = first_match["title"]
            poster = get_poster(movie_title)
            tmdb_date = get_tmdb_release_date(movie_title)
            final_date = tmdb_date if tmdb_date else str(first_match["release_year"])

            col1, col2 = st.columns([1, 2])

            with col1:
                if poster:
                    st.image(poster, use_container_width=True)
                else:
                    st.info("Poster not found")

            with col2:
                st.subheader(movie_title)
                st.write(f"**Type:** {first_match['type']}")
                st.write(f"**Release Date:** {final_date}")
                st.write(f"**Cast:** {first_match['cast']}")
                st.write(f"**Country:** {first_match['country']}")
                st.write(f"**Genres:** {first_match['listed_in']}")

            st.write("### More Matching Results")
            st.dataframe(
                matches[["title", "type", "release_year", "cast"]].head(10),
                use_container_width=True
            )
        else:
            st.warning("Movie not found.")
    st.markdown("</div>", unsafe_allow_html=True)

    # TRENDING
    st.markdown("<div class='section-box'>", unsafe_allow_html=True)
    st.markdown("<div class='card-title'>🔥 Trending Now</div>", unsafe_allow_html=True)

    sample_n = min(6, len(filtered_df))
    trending_titles = filtered_df.sample(sample_n)["title"].tolist() if sample_n > 0 else []

    cols = st.columns(3)
    for i, title in enumerate(trending_titles):
        poster = get_poster(title)
        with cols[i % 3]:
            if poster:
                st.image(poster, use_container_width=True)
            st.markdown(f"<div class='poster-caption'>{title}</div>", unsafe_allow_html=True)
            if st.button("Add Watchlist", key=f"watch_{title}_{i}"):
                if title not in st.session_state.watchlist:
                    st.session_state.watchlist.append(title)
    st.markdown("</div>", unsafe_allow_html=True)

    # WATCHLIST
    st.markdown("<div class='section-box'>", unsafe_allow_html=True)
    st.markdown("<div class='card-title'>❤️ Your Watchlist</div>", unsafe_allow_html=True)

    if not st.session_state.watchlist:
        st.write("No movies added yet.")
    else:
        cols = st.columns(3)
        for i, title in enumerate(st.session_state.watchlist):
            poster = get_poster(title)
            with cols[i % 3]:
                if poster:
                    st.image(poster, use_container_width=True)
                st.markdown(f"<div class='poster-caption'>{title}</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # AI RECOMMENDATION
    st.markdown("<div class='section-box'>", unsafe_allow_html=True)
    st.markdown("<div class='card-title'>🤖 AI Recommendation</div>", unsafe_allow_html=True)

    model_df = df.copy()
    model_df["combined"] = (
        model_df["title"].fillna("") + " " +
        model_df["director"].fillna("") + " " +
        model_df["listed_in"].fillna("")
    )

    vectorizer = CountVectorizer(stop_words="english")
    matrix = vectorizer.fit_transform(model_df["combined"])
    similarity = cosine_similarity(matrix)

    movie_list = sorted(model_df["title"].dropna().unique().tolist())
    selected_movie = st.selectbox("Choose Movie", movie_list)

    if st.button("Recommend Movies"):
        try:
            index = model_df[model_df["title"] == selected_movie].index[0]
            scores = list(enumerate(similarity[index]))
            scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:7]

            cols = st.columns(3)
            for i, item in enumerate(scores[:6]):
                title = model_df.iloc[item[0]]["title"]
                poster = get_poster(title)
                cast = model_df.iloc[item[0]]["cast"]
                year = model_df.iloc[item[0]]["release_year"]

                with cols[i % 3]:
                    if poster:
                        st.image(poster, use_container_width=True)
                    st.markdown(f"**{title}**")
                    st.write(f"Release: {year}")
                    st.write(f"Cast: {cast[:80]}...")
        except Exception:
            st.warning("Recommendation not available.")
    st.markdown("</div>", unsafe_allow_html=True)

    # POPULAR MOVIES GRID
    st.markdown("<div class='section-box'>", unsafe_allow_html=True)
    st.markdown("<div class='card-title'>🎬 Popular Movies</div>", unsafe_allow_html=True)

    sample_n = min(9, len(filtered_df))
    popular_titles = filtered_df.sample(sample_n)["title"].tolist() if sample_n > 0 else []

    cols = st.columns(3)
    for i, title in enumerate(popular_titles):
        poster = get_poster(title)
        with cols[i % 3]:
            if poster:
                st.image(poster, use_container_width=True)
            st.markdown(f"<div class='poster-caption'>{title}</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # CHARTS AT BOTTOM
    st.markdown("<div class='section-box'>", unsafe_allow_html=True)
    st.markdown("<div class='card-title'>📊 Netflix Data Insights</div>", unsafe_allow_html=True)

    pie = px.pie(df, names="type", title="Movies vs TV Shows")
    pie.update_layout(
        paper_bgcolor="#181818",
        plot_bgcolor="#181818",
        font_color="white",
        title_font_color="white"
    )
    st.plotly_chart(pie, use_container_width=True)

    country_counts = df["country"].value_counts().head(10).reset_index()
    country_counts.columns = ["country", "count"]
    country_chart = px.bar(country_counts, x="country", y="count", title="Top Countries")
    country_chart.update_layout(
        paper_bgcolor="#181818",
        plot_bgcolor="#181818",
        font_color="white",
        title_font_color="white"
    )
    st.plotly_chart(country_chart, use_container_width=True)

    year_chart = px.histogram(df, x="release_year", title="Release Year Distribution")
    year_chart.update_layout(
        paper_bgcolor="#181818",
        plot_bgcolor="#181818",
        font_color="white",
        title_font_color="white"
    )
    st.plotly_chart(year_chart, use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

# =========================
# ROUTING
# =========================
if st.session_state.page_mode == "landing":
    show_landing_page()
else:
    show_dashboard()
