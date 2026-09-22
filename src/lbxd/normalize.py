"""Stage 0: turn raw stars into a comparable taste signal.

This is the step that decides how good everything downstream can possibly be, so
it is worth understanding exactly what each number is for.

We compute three different quantities from the same ratings, because they answer
three different questions:

  z-score      "How much did you like this, relative to YOUR OWN bar?"
               z = (rating - your_mean) / your_stdev
               Removes the generous-rater / harsh-rater difference. This is what
               feeds the collaborative filtering, because it is the only version
               that is comparable across people.

  residual     "How much did you like this, relative to EVERYONE ELSE?"
               residual = rating - the film's average rating
               This is the taste fingerprint. "You liked Parasite" is not
               information -- everybody liked Parasite. "You rated Parasite half
               a star under consensus and Burning a full star over" is
               information. Computed later, in match.py, because it needs the
               film's crowd average.

  weight       "How much should this still count?"
               w = 0.5 ** (age_in_years / half_life)
               Taste drifts. A rating from nine years ago should not carry the
               same weight as one from last month when we describe who you are
               *now*. Applies to the taste profile only -- never to the
               already-seen filter.

Why divide by sigma at all, rather than just subtracting the mean? Two reasons.
First, it equalises contribution: a person who uses the full 0.5-5 range would
otherwise swamp every similarity calculation compared to someone who lives
between 3 and 4. Second, it puts your ratings on the same scale as the MovieLens
users' ratings, which is what lets us mix them at all.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import NORMALIZE, NormalizeConfig
from .ingest import LetterboxdExport


@dataclass
class TasteProfile:
    """A member's ratings, normalized and weighted, plus the stats to invert it."""

    films: pd.DataFrame  # title, year, lb_uri, rating, z, weight, is_liked, has_review
    mean: float
    sigma: float  # the floored sigma actually used
    sigma_raw: float  # the observed sigma, before flooring
    reference_date: pd.Timestamp
    config: NormalizeConfig
    warnings: list[str]

    def to_stars(self, z: float | np.ndarray, gain: float | None = None) -> float | np.ndarray:
        """Convert a predicted z-score back into predicted stars.

        This is why we bother being careful about mean and sigma: it lets the
        product say "we think you'll give this 4.2" -- a falsifiable, checkable
        claim -- instead of an invented "97% match".

        `gain` defaults to 1, which keeps this an exact inverse of the z-score:
        to_stars(z_of(r)) == r. That identity matters -- this method is also used
        to put the CROWD's opinion on your scale, and to read your own ratings
        back -- so the calibration gain is NOT applied here by default. Use
        predicted_stars() for a CF prediction, which is the only place the
        attenuation correction belongs.
        """
        g = 1.0 if gain is None else gain
        stars = self.mean + self.sigma * g * np.asarray(z, dtype="float64")
        return np.clip(stars, 0.5, 5.0)

    def predicted_stars(self, pred_z: float | np.ndarray, gain: float | None = None):
        """Turn a CF prediction into stars, correcting for attenuation.

        Separate from to_stars() because a prediction and an observation are not
        the same kind of number. Averaging over neighbours shrinks the spread of
        predictions to roughly half that of real ratings, so a prediction
        converted naively says 3.4 where the truth is 4.1. See CFConfig.pred_gain
        for the measurement.
        """
        from .config import CF

        return self.to_stars(pred_z, CF.pred_gain if gain is None else gain)

    def summary(self) -> str:
        f = self.films
        lines = [
            f"films rated     : {len(f)}",
            f"your mean       : {self.mean:.3f}",
            f"your stdev      : {self.sigma_raw:.3f}"
            + (f"  (floored to {self.sigma:.3f})" if self.sigma != self.sigma_raw else ""),
            f"reference date  : {self.reference_date.date()}",
            f"with reviews    : {int(f['has_review'].sum())}",
            f"liked films     : {int(f['is_liked'].sum())}"
            + ("  (EXCLUDED from signal)" if not self.config.use_likes else "  (included)"),
        ]
        if self.config.recency_half_life_years:
            lines.append(
                f"recency weights : {f['weight'].min():.3f} - {f['weight'].max():.3f} "
                f"(median {f['weight'].median():.3f})"
            )
        top = f.nlargest(5, "z")[["title", "year", "rating", "z"]]
        lines.append("\nyour strongest positives (by z):")
        for _, row in top.iterrows():
            yr = "" if pd.isna(row["year"]) else f" ({int(row['year'])})"
            lines.append(f"  {row['rating']:.1f}  z={row['z']:+.2f}  {row['title']}{yr}")
        bot = f.nsmallest(5, "z")[["title", "year", "rating", "z"]]
        lines.append("\nyour strongest negatives (by z) -- these are signal too:")
        for _, row in bot.iterrows():
            yr = "" if pd.isna(row["year"]) else f" ({int(row['year'])})"
            lines.append(f"  {row['rating']:.1f}  z={row['z']:+.2f}  {row['title']}{yr}")
        if self.warnings:
            lines.append("\nwarnings:")
            lines += [f"  ! {w}" for w in self.warnings]
        return "\n".join(lines)


def build_profile(
    export: LetterboxdExport, config: NormalizeConfig = NORMALIZE
) -> TasteProfile:
    films = export.ratings.copy()
    films = films.dropna(subset=["rating"]).drop_duplicates(subset=["title", "year"], keep="last")

    warnings: list[str] = []

    # ---- centre and scale -------------------------------------------------
    mean = float(films["rating"].mean())
    sigma_raw = float(films["rating"].std(ddof=0))
    sigma = max(sigma_raw, config.sigma_floor)
    if sigma != sigma_raw:
        warnings.append(
            f"your rating spread ({sigma_raw:.3f}) is narrower than the floor "
            f"({config.sigma_floor}); z-scores are being compressed deliberately "
            "so that ordinary rating noise is not read as strong preference"
        )
    films["z"] = (films["rating"] - mean) / sigma

    # ---- recency weighting -------------------------------------------------
    #
    # Reference point is the newest activity in the export, not today's date, so
    # that re-running this next month does not silently decay everything and
    # change your results. Reproducibility beats freshness here.
    dates = pd.to_datetime(films["logged_date"], errors="coerce")
    reference = dates.max()
    if pd.isna(reference):
        reference = pd.Timestamp.today().normalize()
        warnings.append("no usable rating dates; recency weighting disabled")
        films["weight"] = 1.0
    elif config.recency_half_life_years is None:
        films["weight"] = 1.0
    else:
        # A very common real-world trap: people bulk-import their back catalogue
        # when they join, so hundreds of ratings share one date. Recency
        # weighting on that data is not just useless, it is actively misleading --
        # it would treat an entire imported history as equally "current".
        top_share = dates.value_counts(normalize=True).max() if dates.notna().any() else 0.0
        if top_share > 0.25:
            busiest = dates.value_counts().idxmax()
            warnings.append(
                f"{top_share:.0%} of your ratings share a single date "
                f"({busiest.date()}) -- this looks like a bulk import, so recency "
                "weighting is unreliable. Consider recency_half_life_years=None."
            )
        age_years = (reference - dates).dt.total_seconds() / (365.25 * 24 * 3600)
        age_years = age_years.fillna(age_years.median()).clip(lower=0)
        films["weight"] = 0.5 ** (age_years / config.recency_half_life_years)

    # ---- side signals ------------------------------------------------------
    liked_keys = set()
    if not export.likes.empty:
        liked_keys = set(zip(export.likes["title"], export.likes["year"]))
    films["is_liked"] = [
        (t, y) in liked_keys for t, y in zip(films["title"], films["year"])
    ]

    reviewed_keys: set = set()
    if not export.reviews.empty and "review" in export.reviews.columns:
        rev = export.reviews[export.reviews["review"].astype(str).str.strip().ne("")]
        reviewed_keys = set(zip(rev["title"], rev["year"]))
    films["has_review"] = [
        (t, y) in reviewed_keys for t, y in zip(films["title"], films["year"])
    ]

    # Likes are off by default. When enabled they nudge z rather than replacing
    # it, because a like is a weaker and much noisier statement than a rating.
    if config.use_likes:
        films.loc[films["is_liked"], "z"] += config.like_bonus_z

    films = films.reset_index(drop=True)
    return TasteProfile(
        films=films,
        mean=mean,
        sigma=sigma,
        sigma_raw=sigma_raw,
        reference_date=reference,
        config=config,
        warnings=warnings,
    )
