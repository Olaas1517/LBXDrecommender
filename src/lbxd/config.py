"""Central configuration: paths and the tunable knobs of the algorithm.

Every magic number in this project lives here, with a comment explaining what it
does and which direction to move it. Nothing else should hardcode a constant.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# ---------------------------------------------------------------- paths

ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"

ML_DIR = DATA_RAW / "ml-32m"
ML_ZIP = DATA_RAW / "ml-32m.zip"


@dataclass(frozen=True)
class NormalizeConfig:
    """Stage 0: turning raw stars into a comparable taste signal."""

    # Floor on a user's rating standard deviation.
    #
    # WHY: z-scores divide by sigma. Someone who rated 600 films entirely within
    # 3.5-4.0 stars has sigma ~= 0.15, which would inflate ordinary rating noise
    # into z-scores of +/-3 ("this is the greatest film I have ever seen").
    # The floor encodes a prior: we don't believe anyone's true internal scale is
    # tighter than half a star. Raise it to be more conservative about
    # narrow-range raters.
    sigma_floor: float = 0.5

    # Half-life, in years, for recency weighting of the *taste profile*.
    #
    # WHY 3 years: taste drifts, but slowly. A 3-year half-life means a film you
    # rated 3 years ago counts half as much as one from last month, and one from
    # 9 years ago counts an eighth. Set to None to disable decay entirely.
    # NOTE: decay applies to the taste profile only, never to the "already seen"
    # filter -- you have still seen a film you watched in 2011.
    recency_half_life_years: float | None = 3.0

    # Should film likes (likes/films.csv) contribute to the taste signal?
    #
    # WHY default False: likes are a noisy, per-user-inconsistent signal. Many
    # members treat the heart as a bookmark or a social gesture rather than an
    # endorsement. Turn it on and let the eval harness tell you whether it helps
    # on YOUR data -- that is exactly the kind of question the harness exists for.
    use_likes: bool = False

    # If likes are enabled, how much of a z-score bump a like is worth.
    like_bonus_z: float = 0.25


@dataclass(frozen=True)
class MatchConfig:
    """Letterboxd title+year -> MovieLens movieId matching."""

    # How many years of slack when matching titles.
    #
    # WHY 1: Letterboxd and MovieLens sometimes disagree by a year on release
    # date (festival premiere vs general release, or a late-December film). More
    # than 1 year of slack starts matching genuine remakes to each other, which
    # is a much worse error than a miss.
    year_tolerance: int = 1

    # Minimum rapidfuzz score (0-100) to accept a fuzzy title match.
    #
    # WHY 88: below roughly 85 you start pairing sequels and same-franchise
    # entries ("Ocean's Eleven" / "Ocean's Twelve" score ~86). 88 with a year
    # constraint is conservative; misses go to a report file for eyeballing
    # rather than being silently guessed.
    fuzzy_threshold: int = 88


@dataclass(frozen=True)
class CFConfig:
    """Item-item collaborative filtering."""

    # Ignore MovieLens films with fewer than this many ratings.
    #
    # WHY 20: below ~20 ratings a film's rating vector is too sparse for its
    # similarity to anything to be meaningful, and it bloats the matrix. This is
    # the single biggest lever on precompute time and memory. It does cost us
    # coverage of the very deepest obscurities -- an acceptable trade because a
    # film with 8 ratings gives us nothing to reason from anyway.
    min_item_ratings: int = 20

    # Ignore MovieLens users with fewer than this many ratings.
    # WHY 5: users with 1-4 ratings add noise and no co-rating structure.
    min_user_ratings: int = 5

    # Shrinkage constant lambda for similarity confidence.
    #
    # WHY this exists (the most important knob in the project): two films rated
    # by only 3 people in common can show a correlation of 1.0 purely by chance.
    # Uncorrected, these spurious perfect matches dominate the top of every
    # recommendation list and the whole system produces garbage.
    #
    # We multiply each raw similarity by n_ij / (n_ij + lambda), where n_ij is
    # the number of users who rated both films. The shape is what makes it good:
    #   n_ij = lambda      -> factor 0.5  (half-trusted)
    #   n_ij >> lambda     -> factor -> 1 (trusted as-is)
    #   n_ij << lambda     -> factor -> 0 (discarded)
    # So lambda reads directly as "how many co-raters before I half-believe it".
    # This is standard empirical-Bayes shrinkage toward zero.
    shrinkage_lambda: float = 60.0

    # Keep only the top-K most similar neighbours per film.
    #
    # WHY 200: the full 87k x 87k similarity matrix is ~30 GB dense and mostly
    # noise. Truncating to the strongest 200 neighbours per film keeps the signal,
    # makes the matrix sparse enough to hold in RAM, and acts as a second noise
    # filter. Standard practice; 100-500 all work.
    top_k_neighbours: int = 200

    # The novelty dial. Applied at SCORING time, never baked into the matrix.
    #
    # WHY at scoring time: the similarity matrix takes minutes to build and this
    # is the knob you will actually fiddle with, so retuning stays instant.
    #
    #   rank_score = pred_z - popularity_alpha * log10(n_i / median_n)
    #
    # Subtractive in log-popularity space rather than multiplicative, so it stays
    # monotonic and behaves sanely when pred_z is negative. The units are
    # meaningful: it is "how many z-units of predicted rating I will trade away
    # for a film that is 10x less widely seen".
    #   0.0  -> no preference; recommends crowd-pleasers (the documented failure
    #           mode of the popular open-source Letterboxd recommenders)
    #   0.15 -> mild nudge toward the less-seen
    #   0.4  -> aggressive; surfaces real obscurities, and more misses
    popularity_alpha: float = 0.15

    # Minimum total similarity mass before we are willing to make a claim.
    #
    # WHY: a prediction assembled from 2 weak neighbours is a guess wearing a
    # number. Films below this threshold are unrankable rather than silently
    # ranked, so the engine refuses instead of embarrassing itself.
    #
    # MEASURED, not guessed: at 1.0 the harness could only score 48% of held-out
    # liked films, and median percentile on the less-popular quintiles collapsed
    # to 0.28. At 0.1 coverage rose to 83% and Q1-Q3 to ~0.70. The strict setting
    # was not being cautious, it was throwing away half the answers. Raising it
    # does improve MAE (0.83 vs 0.88) but only by declining to answer, which is
    # the wrong trade for a recommender.
    min_support: float = 0.1


@dataclass(frozen=True)
class EvalConfig:
    """Offline evaluation."""

    # Fraction of the user's ratings held out as the test set.
    holdout_frac: float = 0.2

    # A held-out film counts as "relevant" (something we should have found) if
    # the user rated it at least this many sigma above their own mean.
    #
    # WHY 0.5: we are not trying to predict everything they watched, we are
    # trying to predict what they LIKED. Half a sigma above their own average is
    # a defensible line for "this was a good watch for me".
    relevant_z_threshold: float = 0.5

    ks: tuple[int, ...] = (10, 20, 50, 100)

    # Random seed so that tuning runs are comparable to each other.
    seed: int = 20260730


NORMALIZE = NormalizeConfig()
MATCH = MatchConfig()
CF = CFConfig()
EVAL = EvalConfig()
