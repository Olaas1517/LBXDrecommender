"""Fixtures: a miniature MovieLens with a known right answer.

Everything here is synthetic and tiny, so the whole suite runs in seconds with
no downloads. That is deliberate -- a test suite that needs a 227 MB corpus is a
test suite nobody runs.

The world below has two taste clusters that do not overlap:

    items 0-5   "arthouse"   liked by users 0-39
    items 6-11  "blockbuster" liked by users 40-79

and item 12, a universally-adored film everyone rates highly. That last one is
the interesting fixture: it is what separates a recommender from a popularity
chart, because consensus loves it and it tells you nothing about anybody.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from lbxd.movielens import MovieLens

N_ARTHOUSE, N_BLOCKBUSTER = 6, 6
CONSENSUS_ITEM = 12
N_ITEMS = 13


def _items_frame(n_ratings: np.ndarray, mean_rating: np.ndarray) -> pd.DataFrame:
    titles = (
        [f"Arthouse {i}" for i in range(N_ARTHOUSE)]
        + [f"Blockbuster {i}" for i in range(N_BLOCKBUSTER)]
        + ["Universally Adored"]
    )
    return pd.DataFrame(
        {
            "item_idx": np.arange(N_ITEMS, dtype="int32"),
            "movieId": np.arange(1, N_ITEMS + 1, dtype="int32"),
            "clean_title": titles,
            "title": [f"{t} ({1990 + i})" for i, t in enumerate(titles)],
            "year": pd.array([1990 + i for i in range(N_ITEMS)], dtype="Int64"),
            "genres": ["Drama"] * N_ITEMS,
            "imdbId": pd.array([f"{i:07d}" for i in range(N_ITEMS)], dtype="string"),
            "tmdbId": pd.array([str(1000 + i) for i in range(N_ITEMS)], dtype="string"),
            "n_ratings": n_ratings.astype("int32"),
            "mean_rating": mean_rating.astype("float32"),
        }
    )


@pytest.fixture
def mini_ml() -> MovieLens:
    rng = np.random.default_rng(7)
    users, items, ratings = [], [], []

    for u in range(80):
        arthouse_fan = u < 40
        for i in range(N_ITEMS):
            if i == CONSENSUS_ITEM:
                r = 4.5 + rng.normal(0, 0.15)
            elif i < N_ARTHOUSE:
                r = (4.2 if arthouse_fan else 2.0) + rng.normal(0, 0.25)
            else:
                r = (2.0 if arthouse_fan else 4.2) + rng.normal(0, 0.25)
            users.append(u)
            items.append(i)
            ratings.append(float(np.clip(round(r * 2) / 2, 0.5, 5.0)))

    user_idx = np.array(users, dtype="int32")
    item_idx = np.array(items, dtype="int32")
    rating = np.array(ratings, dtype="float32")

    df = pd.DataFrame({"item_idx": item_idx, "rating": rating})
    stats = df.groupby("item_idx")["rating"].agg(["size", "mean"])
    return MovieLens(
        user_idx=user_idx,
        item_idx=item_idx,
        rating=rating,
        items=_items_frame(stats["size"].to_numpy(), stats["mean"].to_numpy()),
    )
