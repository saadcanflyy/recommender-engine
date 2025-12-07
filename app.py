import streamlit as st
st.write("✅ App started")


import streamlit as st
import pandas as pd
from src.recommender import RecommenderEngine

st.set_page_config(page_title="Recommender Engine", layout="wide")

@st.cache_resource
def load_engine():
    return RecommenderEngine()

engine = load_engine()

st.title("🎬 Recommender Engine (CF + Deep Learning)")

user_id = st.text_input("User ID (MovieLens users typically 1..943)", value="1")
k = st.slider("Top-K", 5, 30, 10)
mode = st.selectbox("Mode", ["Surprise (CF)", "Deep (Two-Tower)", "Hybrid"])

alpha = None
if mode == "Hybrid":
    alpha = st.slider("Hybrid alpha (more CF ↔ more Deep)", 0.0, 1.0, 0.6)

if st.button("Recommend"):
    if mode == "Surprise (CF)":
        recs = engine.recommend_surprise(user_id, k=k)
        df = pd.DataFrame(recs, columns=["movie_id", "title", "cf_score"])
    elif mode == "Deep (Two-Tower)":
        recs = engine.recommend_deep(user_id, k=k)
        df = pd.DataFrame(recs, columns=["movie_id", "title", "deep_score"])
    else:
        recs = engine.recommend_hybrid(user_id, k=k, alpha=alpha)
        df = pd.DataFrame(recs, columns=["movie_id", "title", "blended", "cf_score", "deep_score"])

    st.dataframe(df, use_container_width=True)
    

