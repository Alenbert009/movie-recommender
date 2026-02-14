import streamlit as st
import requests

API_BASE =  "https://movie-recommender-1-qz4k.onrender.com" or "http://127.0.0.1:8000"

st.set_page_config(
    page_title="Movie Recommender",
    page_icon="🎬",
    layout="wide"
)

# ----------------------------------------------------------
# Helper Functions
# ----------------------------------------------------------
def get_json(url, params=None):
    try:
        r = requests.get(url, params=params, timeout=10)
        if r.status_code == 200:
            return r.json()
        else:
            st.error(f"API error: {r.status_code} — {r.text}")
            return None
    except Exception as e:
        st.error(f"Request failed: {e}")
        return None


def movie_card(movie, key):
    col = st.container()
    with col:
        if st.button(movie["title"], key=key):
            st.session_state["page"] = "details"
            st.session_state["selected_movie_id"] = movie["tmdb_id"]

        if movie.get("poster_url"):
            st.image(movie["poster_url"], width=180)
        else:
            st.write("(No poster)")

        st.caption(movie.get("release_date", ""))

    return col


# ----------------------------------------------------------
# Sidebar
# ----------------------------------------------------------
st.sidebar.title("🎬 Movie Recommender")

if "page" not in st.session_state:
    st.session_state["page"] = "home"

if "auto_loaded" not in st.session_state:
    st.session_state["auto_loaded"] = False

category = st.sidebar.selectbox(
    "Home Feed Category",
    ["popular", "trending", "top_rated", "upcoming", "now_playing"],
)

limit = st.sidebar.slider("Limit", 1, 50, 20)

if st.sidebar.button("Load Home Feed"):
    st.session_state["page"] = "home"


# ----------------------------------------------------------
# 🔎 SEARCH BAR — MOVED TO TOP
# ----------------------------------------------------------
st.title("🔎 Search Movies")

query = st.text_input("Search for a movie")

if query:
    suggestions = get_json(
        f"{API_BASE}/tmdb/search",
        params={"query": query, "page": 1},
    )

    if suggestions and "results" in suggestions:
        st.subheader("Suggestions")
        cols = st.columns(5)

        for i, m in enumerate(suggestions["results"][:10]):

            tmdb_id = m["id"]
            title = m.get("title") or m.get("name")
            poster_url = (
                f"https://image.tmdb.org/t/p/w500{m.get('poster_path')}"
                if m.get("poster_path")
                else None
            )
            release_date = m.get("release_date")

            with cols[i % 5]:

                # 📌 Clickable movie card with image
                if st.button(title, key=f"sugg_btn_{i}"):
                    st.session_state["page"] = "details"
                    st.session_state["selected_movie_id"] = tmdb_id

                if poster_url:
                    st.image(poster_url, width=180)

                st.caption(release_date if release_date else "")


# ----------------------------------------------------------
# HOME FEED
# ----------------------------------------------------------
if st.session_state.get("page") == "home":

    st.title("🏠 Home Feed")

    if not st.session_state["auto_loaded"]:
        params = {"category": "trending", "limit": limit}
        st.session_state["auto_loaded"] = True
    else:
        params = {"category": category, "limit": limit}

    movies = get_json(f"{API_BASE}/home", params=params)

    if movies:
        cols = st.columns(5)
        for i, m in enumerate(movies):
            with cols[i % 5]:
                movie_card(m, key=f"home_{i}")


# ----------------------------------------------------------
# MOVIE DETAILS + RECOMMENDATIONS
# ----------------------------------------------------------
if st.session_state.get("page") == "details":

    tmdb_id = st.session_state.get("selected_movie_id")

    if tmdb_id is None:
        st.error("No movie selected.")
    else:
        details = get_json(f"{API_BASE}/movie/id/{tmdb_id}")

        if details:

            st.header(details["title"])

            col1, col2 = st.columns([1, 2])

            with col1:
                if details.get("poster_url"):
                    st.image(details["poster_url"], width=300)
                st.write(f"📅 Release: {details.get('release_date', '')}")
                st.write(f"⭐ Rating: {details.get('vote_average', 'N/A')}")

            with col2:
                st.subheader("Overview")
                st.write(details.get("overview", ""))

                st.subheader("Genres")
                genres = ", ".join([g["name"] for g in details.get("genres", [])])
                st.write(genres)

            st.markdown("---")

            st.subheader("🔁 Complete Recommendations")

            bundle = get_json(
                f"{API_BASE}/movie/search",
                params={
                    "query": details["title"],
                    "tfidf_top_n": 12,
                    "genre_limit": 12,
                },
            )

            if bundle:

                st.subheader("💡 TF-IDF Recommendations")
                tfidf_recs = bundle["tfidf_recommendations"]

                cols = st.columns(4)
                for i, item in enumerate(tfidf_recs):
                    card = item.get("tmdb")
                    if not card:
                        continue
                    with cols[i % 4]:
                        movie_card(card, key=f"tfidf_{i}")

                st.subheader("🎭 Genre-based Recommendations")
                genre_recs = bundle["genre_recommendations"]

                cols = st.columns(4)
                for i, m in enumerate(genre_recs):
                    with cols[i % 4]:
                        movie_card(m, key=f"genre_{i}")

        if st.button("⬅ Back"):
            st.session_state["page"] = "home"
