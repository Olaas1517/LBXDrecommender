"""Stage the MovieLens 32M dataset into a form we can compute on quickly.

MovieLens is our collaborative-filtering substrate: 32,000,204 ratings from
200,948 users across 87,585 films, collected to October 2023. It is the only
large ratings corpus legally available to us, and its links.csv hands us a
movieId -> tmdbId mapping, which is how we get from Letterboxd's title+year to a
stable film identity.

Two things happen here:

1. The 836 MB ratings.csv is read once and re-saved as raw numpy arrays. Reading
   the CSV takes ~40 s; loading the arrays takes under a second. We do this once.

2. Users and films are re-indexed to dense 0..N-1 integers. MovieLens ids are
   sparse (movieId 292757 exists, most ids in between do not). Sparse ids would
   force a 292757-column matrix mostly full of nothing. Dense indices make the
   matrices as small as they can be, which is the difference between a similarity
   matrix that fits in RAM and one that does not.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import CF, DATA_PROCESSED, ML_DIR

_TITLE_YEAR = re.compile(r"^(?P<title>.*?)\s*\((?P<year>\d{4})\)\s*$")

RATINGS_NPZ = DATA_PROCESSED / "ml_ratings.npz"
ITEMS_CSV = DATA_PROCESSED / "ml_items.csv"


@dataclass
class MovieLens:
    """Staged MovieLens data, ready for the CF engine.

    user_idx / item_idx / rating are parallel arrays: one entry per rating event.
    items is a table indexed by item_idx with identity + popularity metadata.
    """

    user_idx: np.ndarray  # int32
    item_idx: np.ndarray  # int32
    rating: np.ndarray  # float32
    items: pd.DataFrame  # item_idx, movieId, tmdbId, imdbId, title, year, n_ratings, mean_rating

    @property
    def n_users(self) -> int:
        return int(self.user_idx.max()) + 1

    @property
    def n_items(self) -> int:
        return len(self.items)

    def summary(self) -> str:
        return (
            f"ratings : {len(self.rating):,}\n"
            f"users   : {self.n_users:,}\n"
            f"films   : {self.n_items:,}\n"
            f"tmdb ids: {int(self.items['tmdbId'].notna().sum()):,}"
            f" ({self.items['tmdbId'].notna().mean():.1%})\n"
            f"years   : {int(self.items['year'].min())} - {int(self.items['year'].max())}"
        )


def _load_movie_metadata() -> pd.DataFrame:
    """movies.csv + links.csv, with the year split out of the title.

    MovieLens writes titles as "Toy Story (1995)" and, awkwardly, moves leading
    articles to the end: "Matrix, The (1999)". Both quirks are handled here so
    that match.py has clean text to work with.
    """
    movies = pd.read_csv(ML_DIR / "movies.csv", encoding="utf-8")
    links = pd.read_csv(
        ML_DIR / "links.csv",
        encoding="utf-8",
        dtype={"movieId": "int32", "imdbId": "string", "tmdbId": "string"},
    )

    parsed = movies["title"].astype(str).str.strip().str.extract(_TITLE_YEAR)
    movies["clean_title"] = parsed["title"].fillna(movies["title"].astype(str).str.strip())
    movies["year"] = pd.to_numeric(parsed["year"], errors="coerce").astype("Int64")

    out = movies.merge(links, on="movieId", how="left")
    return out[["movieId", "clean_title", "title", "year", "genres", "imdbId", "tmdbId"]]


def stage(force: bool = False) -> MovieLens:
    """Build (or load) the staged arrays."""
    if RATINGS_NPZ.exists() and ITEMS_CSV.exists() and not force:
        return load()

    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

    print("reading ratings.csv (836 MB, this takes about a minute)...")
    r = pd.read_csv(
        ML_DIR / "ratings.csv",
        encoding="utf-8",
        usecols=["userId", "movieId", "rating"],
        dtype={"userId": "int32", "movieId": "int32", "rating": "float32"},
    )
    print(f"  {len(r):,} raw ratings")

    # --- filter sparse users and films -------------------------------------
    #
    # Done in two passes because the filters interact: dropping thin users can
    # push a film below the item threshold, and vice versa. Two passes gets us
    # essentially all of the benefit; iterating to a true fixed point is not
    # worth the extra scans.
    for pass_no in (1, 2):
        item_counts = r["movieId"].value_counts()
        keep_items = item_counts.index[item_counts >= CF.min_item_ratings]
        r = r[r["movieId"].isin(keep_items)]

        user_counts = r["userId"].value_counts()
        keep_users = user_counts.index[user_counts >= CF.min_user_ratings]
        r = r[r["userId"].isin(keep_users)]
        print(
            f"  pass {pass_no}: {len(r):,} ratings, "
            f"{r['movieId'].nunique():,} films, {r['userId'].nunique():,} users"
        )

    # --- dense re-indexing --------------------------------------------------
    movie_ids = np.sort(r["movieId"].unique())
    user_ids = np.sort(r["userId"].unique())
    item_map = pd.Series(np.arange(len(movie_ids), dtype="int32"), index=movie_ids)
    user_map = pd.Series(np.arange(len(user_ids), dtype="int32"), index=user_ids)

    item_idx = r["movieId"].map(item_map).to_numpy(dtype="int32")
    user_idx = r["userId"].map(user_map).to_numpy(dtype="int32")
    rating = r["rating"].to_numpy(dtype="float32")

    # --- item table ---------------------------------------------------------
    meta = _load_movie_metadata()
    items = pd.DataFrame({"movieId": movie_ids, "item_idx": np.arange(len(movie_ids), dtype="int32")})
    items = items.merge(meta, on="movieId", how="left")

    # Popularity and consensus quality, straight from the data. n_ratings is the
    # popularity signal the novelty dial acts on; mean_rating is the consensus
    # baseline that Stage 0's residual is measured against.
    grp = pd.DataFrame({"item_idx": item_idx, "rating": rating}).groupby("item_idx")["rating"]
    stats = grp.agg(n_ratings="size", mean_rating="mean")
    items = items.merge(stats, on="item_idx", how="left")
    items["n_ratings"] = items["n_ratings"].fillna(0).astype("int32")
    # A movieId present in ratings.csv but absent from movies.csv would leave a
    # null title here, which the matcher cannot index.
    items["clean_title"] = items["clean_title"].fillna("")

    np.savez_compressed(RATINGS_NPZ, user_idx=user_idx, item_idx=item_idx, rating=rating)
    items.to_csv(ITEMS_CSV, index=False, encoding="utf-8")
    print(f"wrote {RATINGS_NPZ.name} and {ITEMS_CSV.name}")

    return MovieLens(user_idx=user_idx, item_idx=item_idx, rating=rating, items=items)


def load() -> MovieLens:
    if not RATINGS_NPZ.exists():
        raise FileNotFoundError(f"{RATINGS_NPZ} missing. Run stage() first.")
    z = np.load(RATINGS_NPZ)
    items = pd.read_csv(ITEMS_CSV, encoding="utf-8", dtype={"imdbId": "string", "tmdbId": "string"})
    return MovieLens(
        user_idx=z["user_idx"], item_idx=z["item_idx"], rating=z["rating"], items=items
    )


if __name__ == "__main__":
    ml = stage()
    print()
    print(ml.summary())
