from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import joblib  # type: ignore
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class CFArtifacts:
    n_movies: int
    n_users: int
    ratings_n: int
    sim_shape: tuple[int, int]


def build_cf_index(
    movies_csv: Path,
    ratings_csv: Path,
    out_dir: Path,
    min_ratings_per_movie: int = 50,
    min_ratings_per_user: int = 20,
    topk_per_movie: int = 200,
) -> CFArtifacts:
    """
    Build an item-item collaborative filtering index (movie-to-movie similarity)
    using cosine similarity over a sparse user-item rating matrix.

    We store a Top-K neighbor list per movie to keep artifacts compact.
    """
    movies = pd.read_csv(movies_csv)
    ratings = pd.read_csv(ratings_csv)

    required_movies_cols = {"movieId", "title"}
    required_ratings_cols = {"userId", "movieId", "rating"}

    if not required_movies_cols.issubset(movies.columns):
        raise ValueError(f"movies.csv must contain {required_movies_cols}. Found: {set(movies.columns)}")

    if not required_ratings_cols.issubset(ratings.columns):
        raise ValueError(f"ratings.csv must contain {required_ratings_cols}. Found: {set(ratings.columns)}")

    # Filter sparse noise (best practice for CF baselines)
    movie_counts = ratings["movieId"].value_counts()
    user_counts = ratings["userId"].value_counts()

    keep_movies = movie_counts[movie_counts >= min_ratings_per_movie].index
    keep_users = user_counts[user_counts >= min_ratings_per_user].index

    ratings_f = ratings[ratings["movieId"].isin(keep_movies) & ratings["userId"].isin(keep_users)].copy()

    # Build contiguous indices
    movie_ids = np.sort(ratings_f["movieId"].unique())
    user_ids = np.sort(ratings_f["userId"].unique())

    movie_id_to_idx = {mid: i for i, mid in enumerate(movie_ids)}
    user_id_to_idx = {uid: i for i, uid in enumerate(user_ids)}

    rows = ratings_f["userId"].map(user_id_to_idx).to_numpy()
    cols = ratings_f["movieId"].map(movie_id_to_idx).to_numpy()
    vals = ratings_f["rating"].to_numpy(dtype=np.float32)

    mat = csr_matrix((vals, (rows, cols)), shape=(len(user_ids), len(movie_ids)))

    # Cosine similarity between items: use item vectors (transpose)
    # Result shape: (n_movies, n_movies)
    sim = cosine_similarity(mat.T, dense_output=False)

    # Build title mapping only for movies included
    movies_small = movies[movies["movieId"].isin(movie_ids)][["movieId", "title"]].copy()
    movies_small["movieIdx"] = movies_small["movieId"].map(movie_id_to_idx)
    movies_small = movies_small.sort_values("movieIdx")
    titles = movies_small["title"].astype(str).tolist()
    movie_id_by_title_lower = {t.lower(): int(mid) for mid, t in zip(movies_small["movieId"], movies_small["title"].astype(str))}

    # Precompute Top-K neighbors per movie (store indices + scores)
    # Using sparse rows keeps it efficient.
    neighbors_idx = []
    neighbors_score = []

    for i in range(sim.shape[0]):
        row = sim.getrow(i)
        if row.nnz == 0:
            neighbors_idx.append([])
            neighbors_score.append([])
            continue

        inds = row.indices
        scores = row.data

        # exclude self
        mask = inds != i
        inds = inds[mask]
        scores = scores[mask]

        if inds.size == 0:
            neighbors_idx.append([])
            neighbors_score.append([])
            continue

        topk = min(topk_per_movie, inds.size)
        top_pos = np.argpartition(scores, -topk)[-topk:]
        top_sorted = top_pos[np.argsort(scores[top_pos])[::-1]]

        neighbors_idx.append(inds[top_sorted].astype(int).tolist())
        neighbors_score.append(scores[top_sorted].astype(float).tolist())

    out_dir.mkdir(parents=True, exist_ok=True)

    meta = {
        "schema": "cf_item_item_v1",
        "movies_csv": str(movies_csv),
        "ratings_csv": str(ratings_csv),
        "min_ratings_per_movie": int(min_ratings_per_movie),
        "min_ratings_per_user": int(min_ratings_per_user),
        "topk_per_movie": int(topk_per_movie),
        "n_movies": int(len(movie_ids)),
        "n_users": int(len(user_ids)),
        "ratings_n": int(len(ratings_f)),
        "movie_ids": movie_ids.astype(int).tolist(),
        "titles": titles,
        "movie_id_by_title_lower": movie_id_by_title_lower,
    }

    (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    joblib.dump(neighbors_idx, out_dir / "neighbors_idx.joblib")
    joblib.dump(neighbors_score, out_dir / "neighbors_score.joblib")

    return CFArtifacts(
        n_movies=int(len(movie_ids)),
        n_users=int(len(user_ids)),
        ratings_n=int(len(ratings_f)),
        sim_shape=(int(len(movie_ids)), int(len(movie_ids))),
    )


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--movies", required=True, help="Path to movies.csv")
    p.add_argument("--ratings", required=True, help="Path to ratings.csv")
    p.add_argument("--out", default="data/processed/index_cf", help="Output directory for artifacts")
    p.add_argument("--min-movie", type=int, default=50)
    p.add_argument("--min-user", type=int, default=20)
    p.add_argument("--topk", type=int, default=200)
    args = p.parse_args()

    artifacts = build_cf_index(
        Path(args.movies),
        Path(args.ratings),
        Path(args.out),
        min_ratings_per_movie=args.min_movie,
        min_ratings_per_user=args.min_user,
        topk_per_movie=args.topk,
    )
    print("Built CF index:", artifacts)
