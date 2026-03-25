from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pickle
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # ✅ All origins allowed (works on Render + any frontend)
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


try:
    df = pd.read_pickle("df.pkl")                          # ✅ Relative path (Render-compatible)

    with open("tfidf_matrix.pkl", "rb") as f:
        tfidf_matrix = pickle.load(f)

    with open("indices.pkl", "rb") as f:
        indices = pickle.load(f)

except Exception as e:
    print("Error loading files:", e)
    df = None
    tfidf_matrix = None
    indices = None



def recommend(movie: str, n: int = 5):
    if df is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    movie_lower = movie.lower()

    matching_movies = [m for m in indices.keys() if movie_lower in m.lower()]

    if not matching_movies:
        raise HTTPException(status_code=404, detail="Movie not found")

    movie_title = matching_movies[0]
    idx = indices[movie_title]

    sim_scores = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()
    similar_idx = sim_scores.argsort()[::-1][1:n+1]

    similar_idx = [i for i in similar_idx if i < len(df)]

    return df['title'].iloc[similar_idx].values.tolist()


# ✅ Routes
@app.get("/")
def home():
    return {"message": "API Running 🚀"}


@app.get("/recommend/{movie_name}")
def get_recommendations(movie_name: str, n: int = 5):
    return {
        "movie": movie_name,
        "recommendations": recommend(movie_name, n)
    }
