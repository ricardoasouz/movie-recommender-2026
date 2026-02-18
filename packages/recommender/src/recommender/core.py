def recommend_dummy(movie: str, k: int = 5) -> list[str]:
    return [f"{movie} (similar #{i})" for i in range(1, k + 1)]
