import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Netflix AI Dashboard", page_icon="🎬", layout="wide")

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
        linear-gradient(to top, rgba(20,20,20,0.98) 8%, rgba(20,20,20,0.55) 40%, rgba(20,20,20,0.88) 100%),
        radial-gradient(circle at top, rgba(229,9,20,0.20), rgba(20,20,20,0.05) 30%, rgba(20,20,20,0.85) 75%);
    padding: 24px 28px;
}

.topbar {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 18px;
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

.lang-btn, .sign-btn {
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

.sign-btn {
    background: #E50914;
    color: white;
}

.hero-title {
    text-align: center;
    margin-top: 24px;
    margin-bottom: 18px;
}

.hero-title h1 {
    font-size: 58px;
    line-height: 1.05;
    margin-bottom: 10px;
    font-weight: 900;
}

.hero-title p {
    font-size: 22px;
    margin-bottom: 18px;
    color: #f2f2f2 !important;
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
    margin-bottom: 8px;
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

.movie-card {
    background: linear-gradient(160deg, #242424 0%, #171717 100%);
    border: 1px solid rgba(255,255,255,0.06);
    border-left: 4px solid #E50914;
    border-radius: 14px;
    padding: 16px 14px;
    min-height: 210px;
    box-shadow: 0 6px 16px rgba(0,0,0,0.28);
    margin-bottom: 10px;
}

.movie-card h4 {
    color: #ffffff !important;
    margin: 0 0 10px 0;
    font-size: 18px;
    line-height: 1.3;
}

.movie-card p {
    margin: 6px 0;
    color: #d8d8d8 !important;
    font-size: 14px;
}

.badge {
    display: inline-block;
    background: #E50914;
    color: white !important;
    font-size: 12px;
    font-weight: 700;
    padding: 4px 8px;
    border-radius: 999px;
    margin-bottom: 8px;
}

.poster-caption {
    text-align: center;
    font-size: 14px;
    margin-top: 6px;
}

.footer-sign {
    margin-top: 10px;
    text-align: center;
    color: #bdbdbd !important;
    font-size: 14px;
    letter-spacing: 0.4px;
}

@media (max-width: 768px) {
    .logo {
        font-size: 38px;
    }
    .hero-title h1 {
        font-size: 34px;
    }
    .hero-title p {
        font-size: 17px;
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
# LOAD DATA
# =========================
try:
    df = pd.read_csv("netflix_titles.csv")
except Exception:
    st.error("netflix_titles.csv same folder mein rakho.")
    st.stop()

df = df.copy()
df["title"] = df["title"].fillna("Unknown")
df["director"] = df["director"].fillna("Not available")
df["cast"] = df["cast"].fillna("Not available")
df["country"] = df["country"].fillna("Unknown")
df["listed_in"] = df["listed_in"].fillna("Unknown")
df["description"] = df["description"].fillna("No description available.")
df["type"] = df["type"].fillna("Unknown")
df["release_year"] = pd.to_numeric(df["release_year"], errors="coerce").fillna(0).astype(int)

# =========================
# HELPERS
# =========================
def short_text(text, n=140):
    text = str(text)
    return text if len(text) <= n else text[:n].rstrip() + "..."

def render_movie_card(title, item_type="", year="", cast="", country="", genre="", description=""):
    st.markdown(f"""
    <div class="movie-card">
        <div class="badge">{item_type if item_type else "NETFLIX"}</div>
        <h4>{title}</h4>
        <p><b>Release:</b> {year}</p>
        <p><b>Cast:</b> {short_text(cast, 85)}</p>
        <p><b>Country:</b> {country}</p>
        <p><b>Genre:</b> {short_text(genre, 65)}</p>
        <p>{short_text(description, 120)}</p>
    </div>
    """, unsafe_allow_html=True)

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
        <div class="hero-title">
            <h1>Unlimited movies, shows,<br>and more</h1>
            <p>Explore Netflix content with smart search, recommendations, and insights.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Featured Titles")

    collage_df = df[df["type"] == "Movie"].copy()
    sample_n = min(12, len(collage_df))
    collage_titles = collage_df.sample(sample_n, random_state=42) if sample_n > 0 else collage_df.head(0)

    cols = st.columns(4)
    for i, (_, row) in enumerate(collage_titles.iterrows()):
        with cols[i % 4]:
            render_movie_card(
                title=row["title"],
                item_type=row["type"],
                year=row["release_year"],
                cast=row["cast"],
                country=row["country"],
                genre=row["listed_in"],
                description=row["description"]
            )

    st.write("")
    c1, c2, c3 = st.columns([1, 1, 1])
    with c2:
        if st.button("Get Started"):
            st.session_state.page_mode = "dashboard"
            st.rerun()

    st.markdown("<div class='footer-sign'>crafted for Omprakash • netflix data experience</div>", unsafe_allow_html=True)

# =========================
# DASHBOARD PAGE
# =========================
def show_dashboard():
    top_left, top_right = st.columns([4, 1])
    with top_left:
        st.markdown("<h1 class='netflix-red'>Netflix AI Dashboard</h1>", unsafe_allow_html=True)
        st.write("Search title, see release date, cast, description and get recommendations.")
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

    # SEARCH
    st.markdown("<div class='section-box'>", unsafe_allow_html=True)
    st.markdown("<div class='card-title'>🔎 Search Movie</div>", unsafe_allow_html=True)

    search_movie = st.text_input("Type a movie name")

    if search_movie:
        matches = filtered_df[
            filtered_df["title"].str.contains(search_movie, case=False, na=False)
        ]

        if not matches.empty:
            first_match = matches.iloc[0]

            col1, col2 = st.columns([1, 2])

            with col1:
                render_movie_card(
                    title=first_match["title"],
                    item_type=first_match["type"],
                    year=first_match["release_year"],
                    cast=first_match["cast"],
                    country=first_match["country"],
                    genre=first_match["listed_in"],
                    description=first_match["description"]
                )

            with col2:
                st.subheader(first_match["title"])
                st.write(f"**Type:** {first_match['type']}")
                st.write(f"**Release Date:** {first_match['release_year']}")
                st.write(f"**Cast:** {first_match['cast']}")
                st.write(f"**Director:** {first_match['director']}")
                st.write(f"**Country:** {first_match['country']}")
                st.write(f"**Genres:** {first_match['listed_in']}")
                st.write(f"**Description:** {first_match['description']}")

            st.write("### More Matching Results")
            st.dataframe(
                matches[["title", "type", "release_year", "cast", "country"]].head(10),
                use_container_width=True
            )
        else:
            st.warning("Movie not found.")
    st.markdown("</div>", unsafe_allow_html=True)

    # TRENDING
    st.markdown("<div class='section-box'>", unsafe_allow_html=True)
    st.markdown("<div class='card-title'>🔥 Trending Now</div>", unsafe_allow_html=True)

    sample_n = min(6, len(filtered_df))
    trending_rows = filtered_df.sample(sample_n, random_state=7) if sample_n > 0 else filtered_df.head(0)

    cols = st.columns(3)
    for i, (_, row) in enumerate(trending_rows.iterrows()):
        with cols[i % 3]:
            render_movie_card(
                title=row["title"],
                item_type=row["type"],
                year=row["release_year"],
                cast=row["cast"],
                country=row["country"],
                genre=row["listed_in"],
                description=row["description"]
            )
            if st.button("Add Watchlist", key=f"watch_{row['title']}_{i}"):
                if row["title"] not in st.session_state.watchlist:
                    st.session_state.watchlist.append(row["title"])
    st.markdown("</div>", unsafe_allow_html=True)

    # WATCHLIST
    st.markdown("<div class='section-box'>", unsafe_allow_html=True)
    st.markdown("<div class='card-title'>❤️ Your Watchlist</div>", unsafe_allow_html=True)

    if not st.session_state.watchlist:
        st.write("No movies added yet.")
    else:
        cols = st.columns(3)
        for i, title in enumerate(st.session_state.watchlist):
            row = df[df["title"] == title].head(1)
            if not row.empty:
                row = row.iloc[0]
                with cols[i % 3]:
                    render_movie_card(
                        title=row["title"],
                        item_type=row["type"],
                        year=row["release_year"],
                        cast=row["cast"],
                        country=row["country"],
                        genre=row["listed_in"],
                        description=row["description"]
                    )
    st.markdown("</div>", unsafe_allow_html=True)

    # AI RECOMMENDATION
    st.markdown("<div class='section-box'>", unsafe_allow_html=True)
    st.markdown("<div class='card-title'>🤖 AI Recommendation</div>", unsafe_allow_html=True)

    model_df = df.copy()
    model_df["combined"] = (
        model_df["title"].fillna("") + " " +
        model_df["director"].fillna("") + " " +
        model_df["listed_in"].fillna("") + " " +
        model_df["cast"].fillna("")
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
                row = model_df.iloc[item[0]]
                with cols[i % 3]:
                    render_movie_card(
                        title=row["title"],
                        item_type=row["type"],
                        year=row["release_year"],
                        cast=row["cast"],
                        country=row["country"],
                        genre=row["listed_in"],
                        description=row["description"]
                    )
        except Exception:
            st.warning("Recommendation not available.")
    st.markdown("</div>", unsafe_allow_html=True)

    # POPULAR MOVIES
    st.markdown("<div class='section-box'>", unsafe_allow_html=True)
    st.markdown("<div class='card-title'>🎬 Popular Movies</div>", unsafe_allow_html=True)

    sample_n = min(9, len(filtered_df))
    popular_rows = filtered_df.sample(sample_n, random_state=21) if sample_n > 0 else filtered_df.head(0)

    cols = st.columns(3)
    for i, (_, row) in enumerate(popular_rows.iterrows()):
        with cols[i % 3]:
            render_movie_card(
                title=row["title"],
                item_type=row["type"],
                year=row["release_year"],
                cast=row["cast"],
                country=row["country"],
                genre=row["listed_in"],
                description=row["description"]
            )
    st.markdown("</div>", unsafe_allow_html=True)

    # CHARTS
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
    st.markdown("<div class='footer-sign'>built in a cinematic style • omprakash edition</div>", unsafe_allow_html=True)

# =========================
# ROUTING
# =========================
if st.session_state.page_mode == "landing":
    show_landing_page()
else:
    show_dashboard()
