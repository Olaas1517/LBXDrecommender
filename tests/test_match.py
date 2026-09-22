"""Title+year -> MovieLens, which is where the bodies are buried."""

from __future__ import annotations

import pandas as pd
import pytest

from lbxd.config import MatchConfig
from lbxd.match import match_films, norm_title


@pytest.fixture
def films_df():
    def make(rows):
        return pd.DataFrame(rows, columns=["title", "year"])
    return make


def _matched_title(result, ml):
    idx = int(result.matched["item_idx"].iloc[0])
    return ml.items.loc[ml.items["item_idx"] == idx, "clean_title"].iloc[0]


# ------------------------------------------------------------------ norm_title


@pytest.mark.parametrize("raw,expected", [
    ("Matrix, The", "the matrix"),
    ("The Matrix", "the matrix"),
    ("Amélie", "amelie"),
    ("Amélie", "amelie"),
    ("Fast & Furious", "fast and furious"),
    ("WALL·E", "wall e"),
    ("Cidade de Deus", "cidade de deus"),
])
def test_article_inversion_and_accents_collapse_to_one_key(raw, expected):
    assert norm_title(raw) == expected


def test_movielens_and_letterboxd_spellings_produce_the_same_key():
    assert norm_title("Matrix, The") == norm_title("The Matrix")
    assert norm_title("Fabuleux destin d'Amélie Poulain, Le") == norm_title(
        "Le Fabuleux destin d'Amelie Poulain"
    )


# ------------------------------------------------------------------ matching


def test_exact_title_and_year(mini_ml, films_df):
    r = match_films(films_df([["Arthouse 0", 1990]]), mini_ml, verbose=False)
    assert len(r.matched) == 1
    assert r.matched["match_method"].iloc[0] == "exact title+year"


def test_year_off_by_one_is_tolerated(mini_ml, films_df):
    """Festival premiere vs general release disagree constantly."""
    r = match_films(films_df([["Arthouse 0", 1991]]), mini_ml, verbose=False)
    assert len(r.matched) == 1
    assert "+/-1" in r.matched["match_method"].iloc[0]


def test_a_unique_title_with_a_wrong_year_is_refused(mini_ml, films_df):
    """The hole this closes: the matcher used to accept ANY title that appeared
    exactly once, year be damned, under the label "exact title, no year check".

    That silently bound post-2023 films -- which MovieLens cannot contain -- to
    whatever older film happened to share their title, and fed a real rating
    into the wrong film's row. A miss is recoverable. A wrong match is a lie
    that propagates into every recommendation downstream.

    "Universally Adored" is used because it is unique AND has no near-neighbour
    titles in the fixture, so the fuzzy pass cannot rescue it and the year guard
    is the only thing under test.
    """
    r = match_films(films_df([["Universally Adored", 2008]]), mini_ml, verbose=False)
    assert len(r.matched) == 0, "a 6-year year gap must not be accepted"
    assert len(r.unmatched) == 1


def test_the_permissive_behaviour_is_still_available_but_labelled(mini_ml, films_df):
    """Turning the guard off must be possible, and must be visible in the
    report rather than hiding among the trustworthy matches."""
    r = match_films(
        films_df([["Universally Adored", 2008]]), mini_ml,
        MatchConfig(require_year_agreement=False), verbose=False,
    )
    assert len(r.matched) == 1
    assert "YEAR MISMATCH" in r.matched["match_method"].iloc[0]


def test_a_missing_year_still_matches_on_a_unique_title(mini_ml, films_df):
    """No year given is a different situation from a year that disagrees."""
    r = match_films(films_df([["Universally Adored", None]]), mini_ml, verbose=False)
    assert len(r.matched) == 1
    assert r.matched["match_method"].iloc[0] == "exact title, no year given"


def test_remakes_are_not_matched_to_each_other():
    """The error this system must never make: 'Ambulance' (2022) is not
    'Ambulance' (2005). A miss is recoverable; a wrong match is a lie."""
    import numpy as np
    from lbxd.movielens import MovieLens

    items = pd.DataFrame({
        "item_idx": np.array([0, 1], dtype="int32"), "movieId": [1, 2],
        "clean_title": ["Ambulance", "Ambulance"],
        "title": ["Ambulance (2005)", "Ambulance (2022)"],
        "year": pd.array([2005, 2022], dtype="Int64"), "genres": ["Drama"] * 2,
        "imdbId": pd.array(["1", "2"], dtype="string"),
        "tmdbId": pd.array(["1", "2"], dtype="string"),
        "n_ratings": np.array([30, 40], dtype="int32"),
        "mean_rating": np.array([3.0, 3.2], dtype="float32"),
    })
    ml = MovieLens(
        user_idx=np.array([0, 0], dtype="int32"),
        item_idx=np.array([0, 1], dtype="int32"),
        rating=np.array([3.0, 3.0], dtype="float32"), items=items,
    )
    r = match_films(pd.DataFrame([["Ambulance", 2022]], columns=["title", "year"]), ml, verbose=False)
    assert len(r.matched) == 1
    assert int(r.matched["item_idx"].iloc[0]) == 1, "must pick the 2022 film"


def test_unmatched_rows_are_reported_not_guessed(mini_ml, films_df):
    r = match_films(
        films_df([["Arthouse 0", 1990], ["A Film That Does Not Exist Anywhere", 1977]]),
        mini_ml, verbose=False,
    )
    assert len(r.matched) == 1 and len(r.unmatched) == 1
    assert r.unmatched["title"].iloc[0] == "A Film That Does Not Exist Anywhere"
    assert r.rate == pytest.approx(0.5)


def test_residual_is_computed_against_the_crowd(mini_ml):
    """residual = your rating - the crowd's average. The taste fingerprint."""
    films = pd.DataFrame([["Arthouse 0", 1990, 5.0]], columns=["title", "year", "rating"])
    r = match_films(films, mini_ml, verbose=False)
    row = r.matched.iloc[0]
    assert row["residual"] == pytest.approx(5.0 - row["mean_rating"], abs=1e-5)


def test_match_summary_counts_add_up(mini_ml, films_df):
    r = match_films(films_df([["Arthouse 0", 1990], ["Nope", 1800]]), mini_ml, verbose=False)
    assert len(r.matched) + len(r.unmatched) == r.n_input


def test_fuzzy_threshold_is_configurable(mini_ml, films_df):
    strict = match_films(
        films_df([["Arthous 0", 1990]]), mini_ml,
        MatchConfig(fuzzy_threshold=99), verbose=False,
    )
    loose = match_films(
        films_df([["Arthous 0", 1990]]), mini_ml,
        MatchConfig(fuzzy_threshold=80), verbose=False,
    )
    assert len(loose.matched) >= len(strict.matched)
