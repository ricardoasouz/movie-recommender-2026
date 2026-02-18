from __future__ import annotations

import json
from pathlib import Path

import joblib  # type: ignore


class ItemItemCFRecommender:
    def __init__(self, artifacts_dir: str | Path):
        artifacts_dir = Path(artifacts_dir)
        self._meta = json.loads((artifacts_dir / "meta.json").read_text(encoding="utf-8"))

        self._titles: list[str] = self._meta["titles"]
        self._movie_ids: list[int] = self._meta["movie_ids"]
        self._movie_id_by_title_lower: dict[str, int] = self._meta["movie_id_by_title_lower"]

        self._neighbors_idx: list[list[int]] = joblib.load(artifacts_dir / "neighbors_idx.joblib")

        # movieId -> movieIdx mapping
        self._movie_idx_by_id = {mid: i for i, mid in enumerate(self._movie_ids)}

    def recommend(self, movie: str, k: int = 5) -> list[str]:
        movie_id = self._movie_id_by_title_lower.get(movie.strip().lower())
        if movie_id is None:
            return []

        idx = self._movie_idx_by_id.get(movie_id)
        if idx is None:
            return []

        neigh = self._neighbors_idx[idx][:k]
        return [self._titles[i] for i in neigh if 0 <= i < len(self._titles)]

    def search_titles(self, q: str, limit: int = 10) -> list[str]:
        q = q.strip().lower()
        if not q:
            return []

        # simple substring search (fast and good enough for now)
        out = []
        for t in self._titles:
            if q in t.lower():
                out.append(t)
                if len(out) >= limit:
                    break
        return out


def recommend_dummy(movie: str, k: int = 5) -> list[str]:
    return [f"{movie} (similar #{i})" for i in range(1, k + 1)]
