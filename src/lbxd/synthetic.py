"""Evaluate against many held-out MovieLens users instead of one real export.

WHY THIS EXISTS

A single Letterboxd export yields roughly 300 held-out liked films across ten
splits. Cut that five ways by popularity and the least-popular quintile holds
n=9. Any knob tuned on n=9 is tuned on noise, and the project's central question
-- "does personalisation actually beat recommending acclaimed films to
everybody?" -- cannot be settled by one person's taste no matter how carefully
the splits are done.

MovieLens users are real people with real rating histories. Treating a few
hundred of them as stand-in members turns n=9 into n=2000+, and answers a
strictly better question: does this beat consensus *for people in general*,
rather than *for this one user, on this one split*.

THE LEAKAGE PROBLEM, AND THE FIX

A MovieLens user's own ratings contributed to the similarity matrix. Scoring
them against it is testing the system on its own training data. The effect of
one user in 200,948 is small -- but "small" is not something you want holding up
a headline number, so build_similarity() takes an `exclude_users` argument and
the harness insists on it. `verify_disjoint` below fails loudly rather than
quietly measuring the wrong thing.

WHAT IT CANNOT TELL YOU

MovieLens users are not Letterboxd users. They skew American, mainstream, and
older, they rate on a different site with a different population, and nobody
arrives there to log Soviet war epics. So this measures the SCORING, which is
population-general, and not the MATCHING or the catalogue coverage, which are
specific to one person's library. The real export remains the check that the
thing works end to end for the person who owns it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .cf import ItemItemCF
from .config import EVAL, NORMALIZE, EvalConfig
from .evaluate import evaluate, popularity_quintiles
from .movielens import MovieLens

HOLDOUT_USERS_NPY = "holdout_users.npy"


def pick_holdout_users(
    ml: MovieLens,
    n: int = 300,
    min_ratings: int = 150,
    max_ratings: int = 1500,
    seed: int = 20260730,
) -> np.ndarray:
    """Choose evaluation subjects that resemble the actual use case.

    The rating-count window is the point. A MovieLens user with 12 ratings is
    not a stand-in for someone who exports a Letterboxd account, and one with
    6,000 is a cataloguer whose profile behaves nothing like a normal member.
    150-1500 brackets the real export (605) on both sides.
    """
    counts = np.bincount(ml.user_idx, minlength=int(ml.user_idx.max()) + 1)
    eligible = np.flatnonzero((counts >= min_ratings) & (counts <= max_ratings))
    if len(eligible) < n:
        raise ValueError(
            f"only {len(eligible)} users have {min_ratings}-{max_ratings} ratings, "
            f"cannot sample {n}"
        )
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(eligible, size=n, replace=False))


def verify_disjoint(ml: MovieLens, users: np.ndarray, sim_built_excluding: np.ndarray | None) -> None:
    """Refuse to report a number that was measured on training data."""
    if sim_built_excluding is None:
        raise ValueError(
            "The similarity matrix was built without excluding these users, so "
            "evaluating on them measures the system against its own training "
            "data. Rebuild with: python -m scripts.build_cf"
        )
    missing = np.setdiff1d(users, np.asarray(sim_built_excluding))
    if len(missing):
        raise ValueError(
            f"{len(missing)} evaluation users were NOT excluded from the "
            "similarity build. Rebuild before trusting this."
        )


def user_frame(
    ml: MovieLens, user: int, sigma_floor: float | None = None
) -> tuple[pd.DataFrame, float, float]:
    """One MovieLens user, shaped exactly like match.py's `matched` frame.

    Same columns the real pipeline produces, so the identical evaluate() runs
    over both and there is no second code path to keep honest.
    """
    if sigma_floor is None:
        sigma_floor = NORMALIZE.sigma_floor
    sel = ml.user_idx == user
    items = ml.item_idx[sel]
    ratings = ml.rating[sel].astype("float64")

    mean = float(ratings.mean())
    sigma = max(float(ratings.std(ddof=0)), sigma_floor)
    df = pd.DataFrame(
        {
            "item_idx": items.astype("int64"),
            "rating": ratings,
            "z": (ratings - mean) / sigma,
            # MovieLens carries no usable per-rating date for our purposes, so
            # recency weighting is simply off here. Flat weights, stated openly,
            # rather than a fabricated timeline.
            "weight": np.ones(len(items), dtype="float64"),
        }
    )
    return df, mean, sigma


@dataclass
class PooledResult:
    n_users: int
    n_observations: int
    observations: pd.DataFrame  # method, quintile, percentile
    rating_error: dict[str, float]

    def table(self, scoreable_only: bool = False) -> pd.DataFrame:
        """Median percentile by method x quintile, plus an overall column.

        `scoreable_only` restricts to films the engine could actually rank.
        Without it, a quintile where nothing is rankable still shows a number,
        produced entirely by where the -inf ties happened to land. That number
        looks like a measurement and is not one.
        """
        obs = self.observations
        if scoreable_only:
            obs = obs[obs["scoreable"]] if "scoreable" in obs.columns else obs
        if obs.empty:
            return pd.DataFrame()
        wide = (
            obs.groupby(["method", "quintile"])["percentile"].median().unstack("quintile")
        )
        wide.insert(0, "ALL", obs.groupby("method")["percentile"].median())
        first = obs["method"].iloc[0]
        counts = obs[obs["method"] == first].groupby("quintile").size()
        wide.loc["n ="] = [int(len(obs) / obs["method"].nunique())] + [
            int(counts.get(q, 0)) for q in wide.columns[1:]
        ]
        if "scoreable" in self.observations.columns:
            full = self.observations
            cov = full[full["method"] == first].groupby("quintile")["scoreable"].mean()
            wide.loc["rankable ="] = [
                float(full[full["method"] == first]["scoreable"].mean())
            ] + [float(cov.get(q, float("nan"))) for q in wide.columns[1:]]
        return wide

    def summary(self) -> str:
        L = [
            f"users evaluated : {self.n_users}",
            f"observations    : {self.n_observations:,} "
            "(held-out liked films x splits, pooled)",
            "",
            "median percentile rank of held-out liked films",
            "(1.00 = ranked top of ~23,000 candidates; 0.50 = coin flip)",
            "'rankable' = fraction the engine had enough support to score at all;",
            "where that is near 0 the row above it is tie-break noise, not a result.",
            "",
            self.table().to_string(float_format=lambda v: f"{v:.3f}"),
            "",
            "RANKABLE FILMS ONLY (the rows above, with the refusals removed)",
            self.table(scoreable_only=True).to_string(float_format=lambda v: f"{v:.3f}"),
        ]
        if self.rating_error:
            L += [
                "",
                "predicted-star accuracy",
                f"    MAE  {self.rating_error.get('mae', float('nan')):.3f} stars"
                f"   (trivial baseline {self.rating_error.get('mae_trivial', float('nan')):.3f})",
            ]
        return "\n".join(L)


def evaluate_users(
    ml: MovieLens,
    cf: ItemItemCF,
    users: np.ndarray,
    n_repeats: int = 2,
    config: EvalConfig = EVAL,
    verbose: bool = True,
) -> PooledResult:
    """Run the standard harness over each user and pool the raw observations.

    Pooling raw percentiles rather than averaging per-user medians is
    deliberate: a user with 40 held-out liked films and one with 4 should not
    count equally toward the answer.
    """
    cuts = popularity_quintiles(ml)
    frames: list[pd.DataFrame] = []
    errs: list[dict[str, float]] = []

    for i, u in enumerate(users):
        df, mean, sigma = user_frame(ml, int(u))
        rep = evaluate(
            df, cf, ml, mean, sigma,
            exclude_item_idx=None, config=config,
            n_repeats=n_repeats, quintile_cuts=cuts,
        )
        if rep.observations is not None:
            frames.append(rep.observations)
        if rep.rating_error:
            errs.append(rep.rating_error)
        if verbose and (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(users)} users")

    obs = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
        columns=["method", "quintile", "percentile"]
    )
    rating_error = {
        k: float(np.mean([e[k] for e in errs if k in e]))
        for k in ("mae", "rmse", "mae_trivial")
    } if errs else {}

    return PooledResult(
        n_users=len(users),
        n_observations=int(len(obs) / max(obs["method"].nunique(), 1)) if len(obs) else 0,
        observations=obs,
        rating_error=rating_error,
    )
