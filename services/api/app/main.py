from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, Query
from recommender.core import ItemItemCFRecommender, recommend_dummy

app = FastAPI(title="Movie Recommender (2026)")

ARTIFACTS_DIR = Path("data/processed/index_cf")
_cf = ItemItemCFRecommender(ARTIFACTS_DIR) if ARTIFACTS_DIR.exists() else None


@app.get("/health")
def health():
    return {
        "status": "ok",
        "cf_index_loaded": _cf is not None,
        "artifacts_dir": str(ARTIFACTS_DIR),
    }


@app.get("/search")
def search(
    q: str = Query(..., min_length=1),
    limit: int = Query(10, ge=1, le=50),
):
    if _cf is None:
        return {"q": q, "limit": limit, "results": []}

    return {"q": q, "limit": limit, "results": _cf.search_titles(q, limit=limit)}


@app.get("/recommend")
def recommend(
    movie: str = Query(..., min_length=1),
    k: int = Query(5, ge=1, le=50),
):
    if _cf is None:
        return {"movie": movie, "k": k, "recommendations": recommend_dummy(movie, k), "mode": "dummy"}

    recs = _cf.recommend(movie, k)
    # Helpful UX: if not found, return suggestions
    if not recs:
        suggestions = _cf.search_titles(movie, limit=10)
        return {
            "movie": movie,
            "k": k,
            "recommendations": [],
            "mode": "cf_item_item_v1",
            "not_found": True,
            "suggestions": suggestions,
        }

    return {"movie": movie, "k": k, "recommendations": recs, "mode": "cf_item_item_v1"}
