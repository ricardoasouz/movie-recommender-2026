from fastapi import FastAPI, Query
from recommender.core import recommend_dummy

app = FastAPI(title="Movie Recommender (2026)")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/recommend")
def recommend(
    movie: str = Query(..., min_length=1),
    k: int = Query(5, ge=1, le=50),
):
    recs = recommend_dummy(movie, k)
    return {"movie": movie, "k": k, "recommendations": recs}
