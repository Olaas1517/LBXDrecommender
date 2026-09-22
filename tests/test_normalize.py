"""Stage 0: stars -> a comparable taste signal."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from lbxd.config import NormalizeConfig
from lbxd.ingest import LetterboxdExport
from lbxd.normalize import build_profile


def _export(ratings: pd.DataFrame, likes=None, reviews=None) -> LetterboxdExport:
    empty = pd.DataFrame(columns=["logged_date", "title", "year", "lb_uri"])
    return LetterboxdExport(
        ratings=ratings, diary=empty.copy(),
        reviews=reviews if reviews is not None else empty.copy(),
        watchlist=empty.copy(), watched=empty.copy(),
        likes=likes if likes is not None else empty.copy(),
    )


def _ratings(values, dates=None, titles=None) -> pd.DataFrame:
    n = len(values)
    return pd.DataFrame({
        "logged_date": pd.to_datetime(dates if dates is not None else ["2024-01-01"] * n),
        "title": titles if titles is not None else [f"Film {i}" for i in range(n)],
        "year": [2000 + i for i in range(n)],
        "lb_uri": [f"u{i}" for i in range(n)],
        "rating": values,
    })


def test_z_scores_are_centred_on_your_own_mean():
    p = build_profile(_export(_ratings([1.0, 2.0, 3.0, 4.0, 5.0])))
    assert p.mean == pytest.approx(3.0)
    assert p.films["z"].mean() == pytest.approx(0.0, abs=1e-9)
    # The person who rated a film 5 when their mean is 3 should be positive.
    assert p.films.loc[p.films["rating"] == 5.0, "z"].iloc[0] > 0


def test_sigma_floor_stops_a_narrow_rater_looking_ecstatic():
    """Someone who rates everything 3.5-4.0 has sigma ~0.15. Dividing by that
    turns ordinary noise into z=+3 ('greatest film I have ever seen')."""
    p = build_profile(_export(_ratings([3.5, 3.5, 4.0, 4.0, 3.5, 4.0])))
    assert p.sigma_raw < 0.5
    assert p.sigma == 0.5, "sigma must be floored"
    assert p.films["z"].abs().max() < 1.0
    assert any("narrower than the floor" in w for w in p.warnings)


def test_to_stars_inverts_the_z_score():
    p = build_profile(_export(_ratings([1.0, 2.0, 3.0, 4.0, 5.0])))
    for z, rating in zip(p.films["z"], p.films["rating"]):
        assert p.to_stars(z) == pytest.approx(rating, abs=1e-6)


def test_predicted_stars_expands_but_to_stars_does_not():
    """A prediction and an observation are different kinds of number.

    to_stars must stay an exact inverse (it is used to read your own ratings and
    to put the crowd's opinion on your scale). predicted_stars applies the
    attenuation correction, because CF predictions come out under-dispersed.
    """
    from lbxd.config import CF

    p = build_profile(_export(_ratings([1.0, 2.0, 3.0, 4.0, 5.0])))
    z = 1.0
    assert p.to_stars(z) == pytest.approx(p.mean + p.sigma * z)
    assert p.predicted_stars(z) == pytest.approx(p.mean + p.sigma * CF.pred_gain * z)
    # A gain above 1 must push a positive prediction further from the mean.
    assert p.predicted_stars(z) > p.to_stars(z)
    assert p.predicted_stars(-z) < p.to_stars(-z)


def test_to_stars_never_leaves_the_letterboxd_scale():
    p = build_profile(_export(_ratings([1.0, 2.0, 3.0, 4.0, 5.0])))
    assert p.to_stars(50.0) == 5.0
    assert p.to_stars(-50.0) == 0.5


def test_recency_weighting_decays_by_half_life():
    dates = ["2024-01-01", "2021-01-01", "2018-01-01"]  # now, -3y, -6y
    p = build_profile(
        _export(_ratings([5.0, 3.0, 1.0], dates=dates)),
        NormalizeConfig(recency_half_life_years=3.0),
    )
    w = p.films["weight"].to_numpy()
    assert w[0] == pytest.approx(1.0, abs=1e-3)
    assert w[1] == pytest.approx(0.5, abs=0.02)
    assert w[2] == pytest.approx(0.25, abs=0.02)


def test_recency_can_be_switched_off():
    p = build_profile(
        _export(_ratings([5.0, 3.0, 1.0], dates=["2024-01-01", "2014-01-01", "2004-01-01"])),
        NormalizeConfig(recency_half_life_years=None),
    )
    assert (p.films["weight"] == 1.0).all()


def test_reference_date_is_the_export_not_today():
    """Re-running next month must not silently change your results."""
    p = build_profile(_export(_ratings([5.0, 3.0], dates=["2020-05-05", "2019-01-01"])))
    assert p.reference_date == pd.Timestamp("2020-05-05")


def test_bulk_import_is_detected_and_warned_about():
    """People import their back catalogue on signup. Hundreds of ratings share
    one date, and recency weighting on that is actively misleading."""
    dates = ["2021-03-03"] * 8 + ["2024-01-01", "2024-02-01"]
    p = build_profile(_export(_ratings([3.0, 4.0] * 5, dates=dates)))
    assert any("bulk import" in w for w in p.warnings)


def test_likes_are_excluded_by_default_and_opt_in():
    r = _ratings([3.0, 4.0, 5.0], titles=["A", "B", "C"])
    likes = pd.DataFrame({
        "logged_date": pd.to_datetime(["2024-01-01"]),
        "title": ["B"], "year": [2001], "lb_uri": ["u1"],
    })
    off = build_profile(_export(r, likes=likes), NormalizeConfig(use_likes=False))
    on = build_profile(_export(r, likes=likes), NormalizeConfig(use_likes=True, like_bonus_z=0.25))
    assert off.films["is_liked"].sum() == 1
    b_off = off.films.loc[off.films["title"] == "B", "z"].iloc[0]
    b_on = on.films.loc[on.films["title"] == "B", "z"].iloc[0]
    assert b_on == pytest.approx(b_off + 0.25)


def test_duplicate_titles_keep_the_latest_rating():
    r = _ratings([2.0, 5.0], titles=["Same", "Same"])
    r["year"] = [1999, 1999]
    p = build_profile(_export(r))
    assert len(p.films) == 1
    assert p.films["rating"].iloc[0] == 5.0
