"""Read a Letterboxd account export.

Letterboxd gives you this from Settings -> Data -> Export your data (free for all
members). You get a ZIP. We read it directly -- no need to unpack it by hand.

The files we care about, and their columns as Letterboxd writes them:

    ratings.csv       Date, Name, Year, Letterboxd URI, Rating
    diary.csv         Date, Name, Year, Letterboxd URI, Rating, Rewatch, Tags,
                      Watched Date
    reviews.csv       Date, Name, Year, Letterboxd URI, Rating, Rewatch, Review,
                      Tags, Watched Date
    watchlist.csv     Date, Name, Year, Letterboxd URI
    watched.csv       Date, Name, Year, Letterboxd URI
    likes/films.csv   Date, Name, Year, Letterboxd URI

Note there is no TMDB or IMDb id anywhere in the export -- only a title, a year,
and a Letterboxd URI. That absence is why match.py exists.

Everything here is written defensively (case-insensitive headers, tolerant of
missing files) because export schemas drift over time and a personal tool that
crashes on a renamed column is worse than useless.
"""

from __future__ import annotations

import io
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

# Canonical internal column names. We rename Letterboxd's headers to these so the
# rest of the codebase never has to think about "Letterboxd URI" vs "uri".
_CANON = {
    "date": "logged_date",
    "name": "title",
    "year": "year",
    "letterboxd uri": "lb_uri",
    "rating": "rating",
    "rewatch": "rewatch",
    "tags": "tags",
    "watched date": "watched_date",
    "review": "review",
}

_FILM_FILES = {
    "ratings": "ratings.csv",
    "diary": "diary.csv",
    "reviews": "reviews.csv",
    "watchlist": "watchlist.csv",
    "watched": "watched.csv",
    "likes": "likes/films.csv",
}


@dataclass
class LetterboxdExport:
    """Everything we managed to read out of one member's export."""

    ratings: pd.DataFrame
    diary: pd.DataFrame
    reviews: pd.DataFrame
    watchlist: pd.DataFrame
    watched: pd.DataFrame
    likes: pd.DataFrame
    source: Path | None = None
    missing: list[str] = field(default_factory=list)

    def summary(self) -> str:
        n_rev = 0
        if not self.reviews.empty and "review" in self.reviews.columns:
            n_rev = int(self.reviews["review"].astype(str).str.strip().ne("").sum())
        lines = [
            f"source          : {self.source}",
            f"rated films     : {len(self.ratings)}",
            f"diary entries   : {len(self.diary)}",
            f"written reviews : {n_rev}",
            f"watchlist       : {len(self.watchlist)}",
            f"watched (total) : {len(self.watched)}",
            f"liked films     : {len(self.likes)}",
        ]
        if self.missing:
            lines.append(f"missing files   : {', '.join(self.missing)}")
        if not self.ratings.empty:
            r = self.ratings["rating"]
            lines += [
                f"rating mean     : {r.mean():.3f}",
                f"rating stdev    : {r.std():.3f}",
                f"rating range    : {r.min()} - {r.max()}",
            ]
        return "\n".join(lines)

    def watched_titles(self) -> pd.DataFrame:
        """Union of everything the member has seen, deduplicated.

        WHY this is its own thing: the "already seen" filter must be built from
        the *union* of watched.csv, diary.csv and ratings.csv, not from ratings
        alone. Plenty of people log films without rating them, and recommending
        a film someone watched in 2014 and didn't bother to rate is a visible,
        trust-destroying error.
        """
        frames = [
            df[["title", "year", "lb_uri"]]
            for df in (self.watched, self.diary, self.ratings, self.reviews)
            if not df.empty
        ]
        if not frames:
            return pd.DataFrame(columns=["title", "year", "lb_uri"])
        allseen = pd.concat(frames, ignore_index=True)
        return allseen.drop_duplicates(subset=["title", "year"]).reset_index(drop=True)


def _canonicalize(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns={c: _CANON.get(str(c).strip().lower(), str(c).strip().lower()) for c in df.columns})

    if "year" in df.columns:
        df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    if "rating" in df.columns:
        # Letterboxd writes half-star ratings as 0.5 .. 5.0 decimals.
        df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    for col in ("logged_date", "watched_date"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    if "rewatch" in df.columns:
        df["rewatch"] = df["rewatch"].astype(str).str.strip().str.lower().eq("yes")
    if "title" in df.columns:
        df["title"] = df["title"].astype(str).str.strip()
    if "lb_uri" in df.columns:
        df["lb_uri"] = df["lb_uri"].astype(str).str.strip()
    return df


def _empty(cols: list[str]) -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series(dtype="object") for c in cols})


def load_export(path: str | Path) -> LetterboxdExport:
    """Load an export from a .zip or from an already-extracted directory."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Get your export from Letterboxd: "
            "Settings -> Data -> Export your data."
        )

    tables: dict[str, pd.DataFrame] = {}
    missing: list[str] = []

    if path.is_dir():

        def read(rel: str) -> pd.DataFrame | None:
            f = path / rel
            if not f.exists():  # tolerate likes/ living at the top level
                f = path / Path(rel).name
            if not f.exists():
                # An export unpacked with its wrapper folder still intact.
                nested = [
                    c / rel for c in path.iterdir() if c.is_dir() and (c / rel).exists()
                ]
                if not nested:
                    return None
                f = nested[0]
            return pd.read_csv(f, encoding="utf-8")

    elif path.suffix.lower() == ".zip":
        zf = zipfile.ZipFile(path)
        # Map lowercased archive names -> real names. Exports have been seen both
        # flat and nested inside a top-level folder, so we match on the suffix as
        # well as exactly.
        names = {n.lower(): n for n in zf.namelist()}

        # Real exports contain deleted/diary.csv and orphaned/reviews.csv
        # alongside the real ones. Both end with "/diary.csv", so a single pass
        # that accepts either an exact or a suffix match picks whichever the ZIP
        # happens to list first -- which silently profiles you on your deleted
        # entries. Exact match therefore wins outright, and the suffix fallback
        # (for exports nested one folder deep) skips these two directories.
        _IGNORED_DIRS = ("deleted/", "orphaned/")

        def read(rel: str) -> pd.DataFrame | None:
            target = rel.lower()
            hit = names.get(target)
            if hit is None:
                candidates = [
                    real
                    for low, real in names.items()
                    if low.endswith("/" + target)
                    and not any(part in low for part in _IGNORED_DIRS)
                ]
                # Shallowest wins: "exportfolder/diary.csv" over any deeper copy.
                hit = min(candidates, key=lambda n: n.count("/")) if candidates else None
            if hit is None:
                return None
            with zf.open(hit) as fh:
                return pd.read_csv(io.TextIOWrapper(fh, encoding="utf-8"))

    else:
        raise ValueError(f"Expected a .zip or a directory, got {path.suffix or 'no extension'}")

    for key, rel in _FILM_FILES.items():
        try:
            df = read(rel)
        except Exception as exc:  # a malformed CSV should not kill the whole load
            print(f"  ! could not parse {rel}: {exc}")
            df = None
        if df is None:
            missing.append(rel)
            tables[key] = _empty(["logged_date", "title", "year", "lb_uri"])
        else:
            tables[key] = _canonicalize(df)

    # Ratings are the one thing we genuinely cannot proceed without.
    if tables["ratings"].empty or "rating" not in tables["ratings"].columns:
        raise ValueError(
            "No usable ratings.csv in the export. Everything downstream is built "
            "from your ratings, so there is nothing to do without it."
        )

    tables["ratings"] = tables["ratings"].dropna(subset=["rating"])

    return LetterboxdExport(source=path, missing=missing, **tables)
