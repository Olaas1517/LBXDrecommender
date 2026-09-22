"""Presentation-layer filters: what is allowed to be *offered* as a film.

Deliberately separate from scoring, and applied after ranking rather than baked
into the similarity matrix. The distinction matters:

  A Sherlock episode carries real collaborative signal. People who rated "His
  Last Vow" highly rated other things in a pattern worth learning from, and
  throwing that away would make the matrix worse. What must not happen is
  offering it to someone as a film to go and watch on Friday night.

So TV keeps contributing to similarity, and gets removed from the output. Same
logic for the other filters here.

Two problems are solved in this module:

1. TV LEAKS IN. MovieLens inherits TV entries from its contributors and nothing
   in its own metadata marks them -- "His Last Vow ()" looks exactly like a film
   with a missing year. IMDb publishes the authoritative answer in
   `title.basics.titleType`, keyed by the imdbId that MovieLens links.csv
   already hands us. Free, no API key, no scraping.

2. NO SAME-WORK DEDUP. Rating one part of Bondarchuk's *War and Peace* returns
   the other three parts as the top three recommendations. They are technically
   the most similar films in the catalogue and completely useless as advice --
   the user has already decided about that work.
"""

from __future__ import annotations

import gzip
import re
import shutil
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from .config import DATA_RAW, DATA_PROCESSED, FILTER, FilterConfig
from .movielens import MovieLens

IMDB_BASICS_URL = "https://datasets.imdbws.com/title.basics.tsv.gz"
IMDB_BASICS_GZ = DATA_RAW / "title.basics.tsv.gz"
TITLE_TYPES_CSV = DATA_PROCESSED / "imdb_title_types.csv"


# ------------------------------------------------------------------ IMDb types


def download_imdb_basics(force: bool = False) -> Path:
    """Fetch IMDb's title.basics dump (~200 MB gz). Their published dataset."""
    if IMDB_BASICS_GZ.exists() and not force:
        return IMDB_BASICS_GZ
    DATA_RAW.mkdir(parents=True, exist_ok=True)
    print(f"downloading {IMDB_BASICS_URL} (~200 MB, once)...")
    tmp = IMDB_BASICS_GZ.with_suffix(".part")
    with urllib.request.urlopen(IMDB_BASICS_URL) as r, open(tmp, "wb") as f:
        shutil.copyfileobj(r, f)
    tmp.rename(IMDB_BASICS_GZ)
    return IMDB_BASICS_GZ


def build_title_types(force: bool = False) -> pd.DataFrame:
    """tconst -> titleType, cached as a small CSV.

    The full dump is ~11M rows and 200 MB compressed; we need two columns of it,
    so it is reduced once and the reduction is what gets kept.
    """
    if TITLE_TYPES_CSV.exists() and not force:
        return pd.read_csv(TITLE_TYPES_CSV, dtype={"imdbId": "string", "titleType": "string"})

    path = download_imdb_basics()
    print("reducing title.basics to (imdbId, titleType)...")
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        df = pd.read_csv(
            fh, sep="\t", usecols=["tconst", "titleType"],
            dtype={"tconst": "string", "titleType": "string"}, na_values="\\N",
        )
    # MovieLens stores the bare digits ("0114709"); IMDb prefixes "tt".
    df["imdbId"] = df["tconst"].str.removeprefix("tt")
    out = df[["imdbId", "titleType"]].dropna()
    out.to_csv(TITLE_TYPES_CSV, index=False, encoding="utf-8")
    print(f"  {len(out):,} titles -> {TITLE_TYPES_CSV.name}")
    return out


def film_mask(ml: MovieLens, config: FilterConfig = FILTER) -> np.ndarray:
    """Boolean over item_idx: True where the entry is a film we can recommend.

    Unknown types are KEPT. A failed or partial IMDb join should cost us a few
    stray TV episodes, not silently empty the catalogue -- a filter that fails
    closed on 23,000 films is a far worse bug than the one it fixes.
    """
    n = len(ml.items)
    if not config.drop_non_films:
        return np.ones(n, dtype=bool)

    types = build_title_types()
    lookup = dict(zip(types["imdbId"].astype(str), types["titleType"].astype(str)))

    imdb = ml.items["imdbId"]
    keep = np.ones(n, dtype=bool)
    denied = set(config.excluded_types())
    for i, raw in enumerate(imdb.to_numpy()):
        if raw is None or (isinstance(raw, float) and np.isnan(raw)) or pd.isna(raw):
            continue
        key = str(raw).strip()
        if key.endswith(".0"):  # a float round-trip through CSV
            key = key[:-2]
        # links.csv zero-pads to 7; IMDb ids past tt9999999 are 8 digits.
        t = lookup.get(key) or lookup.get(key.zfill(7))
        if t is not None and t in denied:
            keep[i] = False
    return keep


def short_mask(ml: MovieLens) -> np.ndarray:
    """True where the entry is a short film. The complement of film_mask's
    shorts question, for `--only-shorts`."""
    types = build_title_types()
    lookup = dict(zip(types["imdbId"].astype(str), types["titleType"].astype(str)))
    out = np.zeros(len(ml.items), dtype=bool)
    for i, raw in enumerate(ml.items["imdbId"].to_numpy()):
        if pd.isna(raw):
            continue
        key = str(raw).strip()
        if key.endswith(".0"):
            key = key[:-2]
        t = lookup.get(key) or lookup.get(key.zfill(7))
        out[i] = t == "short"
    return out


# ------------------------------------------------------------ same-work dedup

# Part markers, which may sit at the END of a title ("War and Peace, Part I")
# or in the MIDDLE of one, followed by that part's own subtitle:
#
#     "War and Peace, Part I: Andrei Bolkonsky"
#     "Voyna i mir III. - Borogyino"
#
# The second form is why both patterns end in `.*$` rather than `$`. Getting
# this wrong is not academic: the recommender put Voyna i mir III and Voyna i
# mir IV in the same six-film list, which is the exact failure this module
# exists to stop.
_PART = re.compile(
    r"[\s,:;-]+(?:part|pt\.?|vol\.?|volume|chapter|book|episode)\s*"
    r"(?:[ivxlcdm]+|\d+|one|two|three|four|five|six|seven|eight|nine|ten)\b.*$",
    flags=re.I,
)
# A bare roman numeral used as a part number: "Voyna i mir III. - Borogyino".
# Requires either end-of-title or a separator introducing a subtitle, so that a
# title merely ENDING in those letters is not mangled.
_TRAILING_ROMAN = re.compile(
    r"[\s,;-]+[ivxlcdm]{1,5}\.?\s*(?:[-–—:]\s*.*)?$", flags=re.I
)
_PAREN = re.compile(r"\([^()]*\)")
# Same thing with the inner text captured. _simplify() strips parentheses,
# so an alternate title has to be handed over WITHOUT its brackets or it is
# erased on the way in.
_PAREN_INNER = re.compile(r"\(([^()]*)\)")
_CUT = re.compile(
    r"[\s,:;-]*\((?:the\s+)?(?:extended|director'?s|theatrical|special|final|"
    r"redux|uncut|unrated)[^)]*\)\s*$",
    flags=re.I,
)
_NONWORD = re.compile(r"[^\w\s]", flags=re.UNICODE)
_WS = re.compile(r"\s+")


def _strip_parts(title: str) -> tuple[str, bool]:
    """Remove a part marker from a title. Returns (stripped, had_marker).

    Shared by work_key and base_titles so the two can never disagree about what
    the base of a multi-part title is -- which they did, silently, and the
    recommender went on offering part III of a work already being watched.
    """
    base = _CUT.sub("", str(title))
    stripped = _PART.sub("", base)
    if stripped == base:
        stripped = _TRAILING_ROMAN.sub("", base)
    had_marker = stripped != base
    # Never strip a title down to nothing: the 2022 film "X" would become an
    # empty key and collide with everything else that strips to empty.
    if had_marker and len(_simplify(stripped)) < 3:
        return base, False
    return stripped, had_marker


def _simplify(title: str) -> str:
    s = _PAREN.sub(" ", str(title)).lower()
    s = _NONWORD.sub(" ", s)
    return _WS.sub(" ", s).strip()


def work_key(clean_title: str, year) -> str:
    """Identity of the WORK, for collapsing its parts into one recommendation.

    Multi-part releases of one work get keyed on the shared base title, because
    that is what they are. Everything else keeps its year in the key, which is
    what stops the two different films called *Ambulance* (2005, 2022) from being
    merged -- a false merge is a much worse error than a missed dedup, since it
    silently deletes a legitimate recommendation.
    """
    stripped, had_part_marker = _strip_parts(clean_title)
    key = _simplify(stripped)
    if had_part_marker:
        return f"work::{key}"
    y = "" if year is None or pd.isna(year) else str(int(year))
    return f"title::{key}::{y}"


def series_key(clean_title: str) -> str | None:
    """A franchise-ish grouping: everything before the first colon.

    Catches "Star Wars: Episode IV", "The Hunger Games: Catching Fire", "Three
    Colors: Blue". Honestly limited -- it cannot see that "Fast Five" belongs
    with "2 Fast 2 Furious", because nothing in the title says so. It is a cap on
    the most obvious kind of list-eating, not a real franchise resolver.
    """
    head, sep, _ = str(clean_title).partition(":")
    if not sep:
        return None
    k = _simplify(head)
    # A one-word head like "Se7en:" is too generic to group on.
    return k if len(k) >= 4 else None


def base_titles(clean_title: str) -> set[str]:
    """Every simplified base title one catalogue entry should answer to.

    MovieLens writes the Bondarchuk adaptation as "War and Peace (Voyna i mir)"
    but its individual parts as "Voyna i mir III. - Borogyino". The shared
    identity is inside the parentheses, so both the outer title and each
    parenthesised alternate are emitted -- otherwise the two never meet and the
    recommender offers you part III of something you are already watching.
    """
    raw = str(clean_title)
    out = {_simplify(_strip_parts(_PAREN.sub(" ", raw))[0])}
    for inner in _PAREN_INNER.findall(raw):
        out.add(_simplify(_strip_parts(inner)[0]))
    return {k for k in out if len(k) >= 3}


def seen_work_keys(ml: MovieLens, seen_item_idx) -> set[str]:
    """Base titles of everything the member has already watched.

    Fed to diversify(), which suppresses a candidate only when that candidate is
    itself a multi-part entry whose base title is in this set. The multi-part
    condition is what keeps this narrow: it means "you are already watching this
    work, you do not need the other parts", not "you have seen a film with a
    similar name".
    """
    import numpy as np

    idx = np.asarray(list(seen_item_idx), dtype=int)
    if not len(idx):
        return set()
    rows = ml.items.iloc[idx]
    keys: set[str] = set()
    for t in rows["clean_title"].fillna(""):
        keys |= base_titles(t)
    return keys


def diversify(
    df: pd.DataFrame,
    n: int,
    config: FilterConfig = FILTER,
    exclude_works: set[str] | None = None,
) -> pd.DataFrame:
    """Walk a ranked frame best-first, keeping one entry per work and capping
    each series. Returns the first `n` survivors, in rank order."""
    if df.empty:
        return df
    if not config.dedupe_same_work and not config.max_per_series:
        return df.head(n)

    seen_works: set[str] = set()
    series_count: dict[str, int] = {}
    keep_rows = []
    for pos, row in enumerate(df.itertuples(index=False)):
        title = getattr(row, "clean_title", "") or ""
        year = getattr(row, "year", None)

        if config.dedupe_same_work:
            wk = work_key(title, year)
            if wk in seen_works:
                continue
            # A multi-part entry whose work the member has already started.
            if wk.startswith("work::") and exclude_works:
                if base_titles(title) & exclude_works:
                    continue
            seen_works.add(wk)

        if config.max_per_series:
            sk = series_key(title)
            if sk is not None:
                if series_count.get(sk, 0) >= config.max_per_series:
                    continue
                series_count[sk] = series_count.get(sk, 0) + 1

        keep_rows.append(pos)
        if len(keep_rows) >= n:
            break
    return df.iloc[keep_rows]
