"""Match Letterboxd export rows to MovieLens films (and thus to TMDB ids).

This module exists because of one gap in the data: the Letterboxd export contains
no film identifiers. Only a title, a year, and a Letterboxd URI. MovieLens has
movieId and tmdbId. So we have to join on text, which is always where the bodies
are buried.

Everything that makes this hard, and what we do about it:

  Article inversion   MovieLens writes "Matrix, The (1999)"; Letterboxd writes
                      "The Matrix". We rotate trailing articles to the front on
                      both sides so the two forms collide.

  Alternate titles    MovieLens embeds them in parentheses:
                      "Amelie (Fabuleux destin d'Amelie Poulain, Le)". We index
                      the outer title AND the parenthesised one as separate keys.

  Accents             "Amélie" vs "Amelie". Stripped via Unicode decomposition.

  Year disagreement   Festival premiere vs general release, or a late-December
                      film. We allow +/- 1 year (config.year_tolerance).

  Remakes             The reason we do NOT allow more year slack. "Ambulance"
                      (2022) and "Ambulance" (2005) are different films, and
                      matching them to each other is a far worse error than
                      failing to match at all. Year is our only guard, so it stays
                      tight.

Match order is strictest-first, and every match records HOW it was made so that a
bad match rate can be diagnosed rather than guessed at. Anything unmatched is
written to a report file for eyeballing rather than being silently fuzzed into
the nearest thing.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process

from .config import MATCH, MatchConfig
from .movielens import MovieLens

# Trailing articles MovieLens rotates to the end, across the languages it covers.
_ARTICLES = (
    "the", "a", "an", "le", "la", "les", "l'", "il", "el", "los", "las",
    "der", "die", "das", "ein", "eine", "de", "het", "en", "et", "os", "as",
)
_PAREN = re.compile(r"\(([^()]*)\)")
_NONWORD = re.compile(r"[^\w\s]", flags=re.UNICODE)
_WS = re.compile(r"\s+")


def _strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))


def norm_title(s: str) -> str:
    """Canonical form used as the join key on both sides."""
    if not isinstance(s, str):
        return ""
    s = _strip_accents(s).lower().strip()
    s = s.replace("&", " and ")
    # Rotate a trailing article to the front: "matrix, the" -> "the matrix".
    if "," in s:
        head, _, tail = s.rpartition(",")
        if tail.strip() in _ARTICLES:
            s = f"{tail.strip()} {head.strip()}"
    s = _NONWORD.sub(" ", s)
    return _WS.sub(" ", s).strip()


_AKA = re.compile(r"^\s*(a\.k\.a\.?|aka)\s*", flags=re.I)


def _drop_leading_article(key: str) -> str:
    """'the man with the movie camera' -> 'man with the movie camera'.

    Indexed as an extra key so that a leading article present on one side and
    absent on the other does not cost us the match. Letterboxd and MovieLens
    disagree about this constantly.
    """
    first, _, rest = key.partition(" ")
    return rest if rest and first in _ARTICLES else key


def _keys_for(text: str) -> list[str]:
    """Every normalized key one title string should be findable under."""
    k = norm_title(text)
    if not k:
        return []
    return list(dict.fromkeys([k, _drop_leading_article(k)]))


def _title_variants(clean_title: str) -> list[str]:
    """All the normalized strings a MovieLens film should be findable under."""
    if not isinstance(clean_title, str) or not clean_title.strip():
        return []
    variants: list[str] = []
    outer = _PAREN.sub("", clean_title).strip()
    variants += _keys_for(outer)
    for inner in _PAREN.findall(clean_title):
        inner = inner.strip()
        # Parenthesised alternates carry real titles, but often behind an "a.k.a."
        # prefix: "Seven (a.k.a. Se7en)". Strip the prefix, keep the title --
        # otherwise the key becomes "a k a se7en" and never matches anything.
        inner = _AKA.sub("", inner).strip()
        if inner:
            variants += _keys_for(inner)
    variants += _keys_for(clean_title)
    return [v for v in dict.fromkeys(variants) if v]


@dataclass
class MatchResult:
    matched: pd.DataFrame  # input rows + item_idx, movieId, tmdbId, ml_year, match_method
    unmatched: pd.DataFrame
    n_input: int

    @property
    def rate(self) -> float:
        return len(self.matched) / self.n_input if self.n_input else 0.0

    def summary(self) -> str:
        lines = [
            f"input rows : {self.n_input}",
            f"matched    : {len(self.matched)} ({self.rate:.1%})",
            f"unmatched  : {len(self.unmatched)}",
        ]
        if not self.matched.empty:
            lines.append("by method:")
            for method, n in self.matched["match_method"].value_counts().items():
                lines.append(f"  {method:<22} {n}")
        return "\n".join(lines)


@dataclass
class TitleIndex:
    """Lookup structures over the MovieLens catalogue.

    by_key      normalized title -> [(item_idx, year), ...]
    keys_by_year  year -> the keys of films released that year

    keys_by_year exists to make fuzzy matching safe. The obvious implementation --
    fuzzy-match against every title, then check the year -- is subtly broken: if
    the highest-scoring title globally has the wrong year, you discard it and
    never see the correct lower-scoring candidate. Restricting the candidate pool
    to the right years FIRST fixes that, and makes the search ~40x faster as a
    bonus.
    """

    by_key: dict[str, list[tuple[int, int]]]
    keys_by_year: dict[int, list[str]]


def build_index(ml: MovieLens) -> TitleIndex:
    by_key: dict[str, list[tuple[int, int]]] = {}
    keys_by_year: dict[int, set[str]] = {}
    for item_idx, clean, year in zip(
        ml.items["item_idx"].to_numpy(),
        # fillna before astype: pandas 3.0 keeps nulls as NA through astype(str)
        # rather than turning them into the string "nan".
        ml.items["clean_title"].fillna("").astype(str).to_numpy(),
        ml.items["year"].to_numpy(),
    ):
        y = -9999 if pd.isna(year) else int(year)
        for v in _title_variants(clean):
            by_key.setdefault(v, []).append((int(item_idx), y))
            keys_by_year.setdefault(y, set()).add(v)
    return TitleIndex(
        by_key=by_key,
        keys_by_year={y: sorted(ks) for y, ks in keys_by_year.items()},
    )


def match_films(
    films: pd.DataFrame,
    ml: MovieLens,
    config: MatchConfig = MATCH,
    verbose: bool = True,
) -> MatchResult:
    """Match a table with `title` and `year` columns to MovieLens items."""
    index = build_index(ml)

    rows: list[dict] = []
    misses: list[dict] = []

    for rec in films.to_dict("records"):
        raw_title = rec.get("title")
        year = rec.get("year")
        year = None if year is None or pd.isna(year) else int(year)
        user_keys = _keys_for(str(raw_title))

        hit: tuple[int, int] | None = None
        method = ""

        # --- pass 1: exact key match, strictest year rule first --------------
        for key in user_keys:
            candidates = index.by_key.get(key)
            if not candidates:
                continue
            if year is not None:
                exact = [c for c in candidates if c[1] == year]
                if exact:
                    hit, method = exact[0], "exact title+year"
                    break
                near = [c for c in candidates if abs(c[1] - year) <= config.year_tolerance]
                if near:
                    near.sort(key=lambda c: abs(c[1] - year))
                    hit, method = near[0], "exact title, year +/-1"
                    break
            if year is None or len(candidates) == 1:
                # Unique title match with no year to check against: accept it.
                hit, method = candidates[0], "exact title, no year check"
                break

        # --- pass 2: fuzzy, but only within the plausible year window --------
        if hit is None and year is not None and user_keys:
            pool: list[str] = []
            for y in range(year - config.year_tolerance, year + config.year_tolerance + 1):
                pool.extend(index.keys_by_year.get(y, ()))
            if pool:
                key = user_keys[0]

                # token_sort_ratio: robust to word order and small edits.
                # Catches "Man with a Movie Camera" vs "Man with the Movie Camera, The".
                best = process.extractOne(key, pool, scorer=fuzz.token_sort_ratio)
                if best and best[1] >= config.fuzzy_threshold:
                    hit, method = index.by_key[best[0]][0], f"fuzzy sort {best[1]:.0f}"

                # token_set_ratio: scores 100 when one title's words are a SUBSET
                # of the other's. This is what catches "Star Wars" ->
                # "Star Wars: Episode IV - A New Hope".
                #
                # It is also dangerous for exactly that reason -- "Alien" is a
                # subset of "Alien Resurrection" -- so we only allow it when the
                # year matches EXACTLY. Year equality is doing all the safety work
                # here, which is why the looser scorer is acceptable.
                if hit is None:
                    exact_year_pool = index.keys_by_year.get(year, ())
                    if exact_year_pool:
                        best = process.extractOne(key, exact_year_pool, scorer=fuzz.token_set_ratio)
                        if best and best[1] >= 95:
                            cands = [c for c in index.by_key[best[0]] if c[1] == year]
                            if cands:
                                hit, method = cands[0], f"subset, exact year {best[1]:.0f}"

        if hit is None:
            misses.append(rec)
        else:
            out = dict(rec)
            out["item_idx"] = hit[0]
            out["ml_year"] = hit[1]
            out["match_method"] = method
            rows.append(out)

    matched = pd.DataFrame(rows)
    if not matched.empty:
        cols = ml.items[["item_idx", "movieId", "tmdbId", "n_ratings", "mean_rating", "clean_title"]]
        matched = matched.merge(cols, on="item_idx", how="left")
        # Consensus residual: how far above/below the crowd you rated it. This is
        # the taste fingerprint described in normalize.py, and it can only be
        # computed here, because it needs the film's crowd average.
        if "rating" in matched.columns:
            matched["residual"] = matched["rating"] - matched["mean_rating"]

    result = MatchResult(
        matched=matched,
        unmatched=pd.DataFrame(misses),
        n_input=len(films),
    )
    if verbose:
        print(result.summary())
    return result
