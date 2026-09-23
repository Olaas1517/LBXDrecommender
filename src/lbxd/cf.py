"""Item-item collaborative filtering: the retrieval engine.

=============================================================================
WHY ITEM-ITEM, AND NOT THE OBVIOUS ALTERNATIVES
=============================================================================

The job: given your ~440 matched ratings, score all ~23,000 other films.

Option A -- user-user CF ("find people like you, see what they liked").
    Requires searching 200,948 users at query time, and gives no reusable
    precomputed structure. Slow, and it does not degrade gracefully.

Option B -- matrix factorisation (SVD / ALS). This is what almost every
    open-source Letterboxd recommender uses, and it is why they behave the way
    they do. Two problems, one fatal:
      1. You are not in the training set, so your latent vector has to be
         "folded in" after the fact. Workable, but lossy.
      2. Latent factors are a compression. The leading factors capture
         mainstream-vs-arthouse and blockbuster-vs-obscure; genuine idiosyncrasy
         lives in the tail that gets discarded. For a contrarian rater, that is
         precisely the information you cannot afford to throw away. The published
         limitation of the most popular scraped-data Letterboxd recommender is
         that it "tends to recommend very popular movies often, regardless of an
         individual user's taste" -- that is this failure mode.

Option C -- item-item CF (what we do). Precompute film-to-film similarity once.
    Scoring a new person is then a sparse matrix multiply against their own
    ratings: milliseconds, no training, no folding in, no GPU. And critically it
    is *interpretable* -- every recommendation decomposes into "because you rated
    X, Y and Z the way you did", which is exactly the input the explanation layer
    needs. That interpretability is not a nice-to-have; it is the product.

=============================================================================
THE SIMILARITY FORMULA, ONE PIECE AT A TIME
=============================================================================

Start from the naive version and fix it three times.

  v1. cosine similarity between two films' raw rating vectors.
      Broken: dominated by rater generosity. A harsh rater and a kind rater who
      have identical taste look different, and two films both rated by many kind
      raters look similar for no good reason.

  v2. centre each user's ratings first (subtract their mean), then cosine.
      This is "adjusted cosine" (Sarwar et al. 2001), and centring by USER rather
      than by item is the empirically better choice for item-item CF, because
      rater-scale bias is the dominant nuisance factor. We go one step further and
      also divide by each user's standard deviation, i.e. full z-scores, so that a
      user who uses the whole 0.5-5 range does not contribute more to every
      similarity than one who lives between 3 and 4.

  v3. shrink by co-rating support.  <-- the single most important line of code
      Two films rated by only 3 people in common can correlate at 1.0 by pure
      chance. Uncorrected, these spurious perfect matches flood the top of every
      recommendation list, and the system produces confident garbage. So:

          sim = cosine * n_ij / (n_ij + lambda)

      where n_ij is how many users rated both films. Read the shape of it:
          n_ij == lambda   -> multiplied by 0.5   (half-believed)
          n_ij >> lambda   -> multiplied by ~1    (believed)
          n_ij << lambda   -> multiplied by ~0    (discarded)
      So lambda literally means "how many co-raters before I half-trust this".
      This is standard empirical-Bayes shrinkage toward zero.

      A useful side effect: it makes the min_item_ratings filter nearly free. A
      film with 15 ratings can have at most n_ij = 15 with anything, so it is
      automatically shrunk to at most 15/(15+60) = 0.2 of its raw similarity. Thin
      films cannot surface even if we let them in.

  v4. keep only each film's top-K neighbours.
      The full 23,350 x 23,350 matrix is mostly near-zero noise. Truncating to the
      strongest 200 per film keeps the signal, makes it fit comfortably in RAM,
      and acts as a second noise filter.

=============================================================================
THE SCORING FORMULA
=============================================================================

For candidate film j and your rated films i (with z-scores z_i, recency w_i):

                 sum_i  sim(i,j) * z_i * w_i
    pred_z(j) = --------------------------------
                 sum_i  |sim(i,j)| * w_i

Why divide? Without the denominator, a film similar to *many* of your rated films
scores highly just by having more contributors -- popularity bias sneaking back in
through the side door. Dividing turns the sum into a weighted average of your own
z-scores, which means the output has a real interpretation: "how far above or
below your personal mean you are predicted to rate this". That is what makes the
honest star prediction possible:

    predicted_stars = your_mean + your_sigma * pred_z

Note that z_i is signed. Films similar to things you HATED get pushed down, not
merely left out. This is how your negative ratings do real work -- and for a rater
whose mean is 2.6, most of the ratings are negative, so this matters a lot.

The denominator doubles as a confidence measure: it is the total similarity mass
behind the claim. Two weak neighbours produce a number, but not a claim worth
making, so low-support predictions are flagged rather than shown.

=============================================================================
THE NOVELTY DIAL
=============================================================================

    rank_score = pred_z - penalty * log10(n_ratings / median_n_ratings)

Separate from pred_z, deliberately. pred_z stays an honest rating prediction we
can check against reality; rank_score is what we sort by. The penalty is in
z-units per 10x of popularity, so penalty=0.15 reads as "I will give up 0.15
z-units of predicted rating for a film that is 10x less widely seen". Subtractive
in log space rather than multiplicative, so it behaves sanely for negative
predictions and stays monotonic.

Applied at scoring time, never baked into the matrix -- the matrix takes minutes
to build and this is the knob you will actually want to fiddle with.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from .config import CF, DATA_PROCESSED, CFConfig
from .movielens import MovieLens

SIM_NPZ = DATA_PROCESSED / "item_similarity.npz"
ITEM_STATS_NPZ = DATA_PROCESSED / "item_stats.npz"


# ---------------------------------------------------------------- z-scoring


def user_zscores(ml: MovieLens, sigma_floor: float = 0.5) -> np.ndarray:
    """Per-rating z-scores, computed against each rater's own mean and spread.

    bincount is used rather than a groupby because at 31.7M rows the difference
    is roughly a minute versus a second.
    """
    n_users = int(ml.user_idx.max()) + 1
    counts = np.bincount(ml.user_idx, minlength=n_users).astype("float64")
    sums = np.bincount(ml.user_idx, weights=ml.rating, minlength=n_users)
    means = sums / np.maximum(counts, 1)

    # var = E[x^2] - E[x]^2, computed in one more pass rather than materialising
    # per-user arrays.
    sq_sums = np.bincount(ml.user_idx, weights=ml.rating.astype("float64") ** 2, minlength=n_users)
    var = np.maximum(sq_sums / np.maximum(counts, 1) - means**2, 0.0)
    sigmas = np.maximum(np.sqrt(var), sigma_floor)

    return ((ml.rating - means[ml.user_idx]) / sigmas[ml.user_idx]).astype("float32")


def item_z_means(
    ml: MovieLens, z: np.ndarray | None = None, config: CFConfig = CF
) -> np.ndarray:
    """Each film's average z-score across all raters: the consensus baseline.

    This is the quantity the old scorer was missing. `mean_rating` in items.csv
    is the average in raw stars, which mixes the film's quality together with the
    generosity of whoever happened to rate it -- popular family films look better
    than they are because kind raters rate them. Averaging z instead removes the
    rater from the number, leaving how good the film is *relative to the other
    films those same people rated*.

    Shrunk toward 0 by co-rating count, for the same reason similarities are: a
    film with 21 ratings has a noisy mean, and without shrinkage the thinnest
    films in the catalogue supply the most extreme consensus values and take over
    whichever end of the ranking the beta dial favours. Zero is the right
    shrinkage target because z is centred per user, so the global mean z is 0 by
    construction -- "I know nothing about this film" and "this film is exactly
    average" are the same statement.
    """
    if z is None:
        z = user_zscores(ml)
    n_items = ml.n_items
    counts = np.bincount(ml.item_idx, minlength=n_items).astype("float64")
    sums = np.bincount(ml.item_idx, weights=z.astype("float64"), minlength=n_items)
    raw = sums / np.maximum(counts, 1.0)
    shrunk = raw * (counts / (counts + config.consensus_shrinkage_lambda))
    return shrunk.astype("float32")


# ---------------------------------------------------------------- precompute


def build_similarity(
    ml: MovieLens,
    config: CFConfig = CF,
    block_size: int = 512,
    verbose: bool = True,
    exclude_users: np.ndarray | None = None,
) -> sp.csr_matrix:
    """Precompute the shrunk, top-K-truncated item-item similarity matrix.

    `exclude_users` drops those raters' ratings before building. It exists for
    the evaluation harness: scoring a MovieLens user against a matrix their own
    ratings helped build is measuring the system against its own training data.
    The effect of any single user out of 200,948 is tiny, but "tiny" is not an
    argument you want load-bearing under a headline number, and excluding them
    costs nothing.
    """
    z = user_zscores(ml)
    n_items = ml.n_items
    n_users = int(ml.user_idx.max()) + 1

    item_idx, user_idx = ml.item_idx, ml.user_idx
    if exclude_users is not None and len(exclude_users):
        keep = ~np.isin(user_idx, np.asarray(exclude_users, dtype="int64"))
        if verbose:
            print(
                f"  excluding {len(np.unique(exclude_users)):,} held-out users "
                f"({(~keep).sum():,} ratings) from the similarity build"
            )
        z, item_idx, user_idx = z[keep], item_idx[keep], user_idx[keep]

    # Items x users, values = z-scores. CSR so that row blocks are cheap to slice.
    Z = sp.csr_matrix(
        (z, (item_idx, user_idx)), shape=(n_items, n_users), dtype="float32"
    )
    # Binary twin, for counting co-raters.
    B = sp.csr_matrix(
        (np.ones(len(z), dtype="float32"), (item_idx, user_idx)),
        shape=(n_items, n_users),
        dtype="float32",
    )

    if config.item_centered:
        # Subtract each film's own mean z from its OBSERVED entries only.
        #
        # Observed-only is what keeps this sparse and what makes it a Pearson
        # correlation over co-raters rather than an assertion about the millions
        # of people who never saw the film. The raw per-row mean is used rather
        # than the shrunk consensus, because centring is only correct if the
        # centred row actually has zero mean.
        counts = np.diff(Z.indptr).astype("float64")
        sums = np.asarray(Z.sum(axis=1)).ravel().astype("float64")
        row_mean = np.divide(sums, counts, out=np.zeros(len(counts)), where=counts > 0)
        Z = Z.copy()
        Z.data = (Z.data - np.repeat(row_mean, np.diff(Z.indptr))).astype("float32")
        if verbose:
            print(f"  item-centred: row means removed (sd {row_mean.std():.3f})")

    # L2-normalise each item row so that a dot product IS the cosine.
    norms = np.sqrt(np.asarray(Z.multiply(Z).sum(axis=1))).ravel()
    norms[norms == 0] = 1.0
    Zn = sp.diags(1.0 / norms).dot(Z).tocsr()

    ZnT = Zn.T.tocsc()
    BT = B.T.tocsc()

    rows_out, cols_out, vals_out = [], [], []
    k = config.top_k_neighbours

    n_blocks = (n_items + block_size - 1) // block_size
    for bi, start in enumerate(range(0, n_items, block_size)):
        stop = min(start + block_size, n_items)

        # Dense blocks: (block x n_items). At block_size=512 and 23,350 items
        # that is ~48 MB each, which is the whole reason for blocking.
        sim = (Zn[start:stop] @ ZnT).toarray()
        cnt = (B[start:stop] @ BT).toarray()

        # --- the shrinkage step ---
        #
        # maximum(..., 1) guards lambda=0, where a pair with no co-raters gives
        # 0/0. NaN would then pass the `v != 0` filter below (NaN != 0 is True)
        # and land in the matrix, where it silently poisons every prediction that
        # touches it. Sweeping lambda to 0 to see the unshrunk behaviour is a
        # reasonable thing to want to do, so it should not corrupt the build.
        denom = cnt + config.shrinkage_lambda
        np.multiply(sim, cnt / np.maximum(denom, 1e-9), out=sim)

        # A film is trivially its own best neighbour; remove it or every
        # recommendation is just the input back again.
        for local, glob in enumerate(range(start, stop)):
            sim[local, glob] = 0.0

        # Negative similarities are kept (films that anti-correlate are real
        # signal for a contrarian rater), so rank by magnitude.
        mag = np.abs(sim)
        kk = min(k, sim.shape[1] - 1)
        top = np.argpartition(-mag, kk, axis=1)[:, :kk]

        for local in range(sim.shape[0]):
            cols = top[local]
            v = sim[local, cols]
            keep = (v != 0) & np.isfinite(v)
            cols, v = cols[keep], v[keep]
            rows_out.append(np.full(len(cols), start + local, dtype="int32"))
            cols_out.append(cols.astype("int32"))
            vals_out.append(v.astype("float32"))

        if verbose and (bi % 5 == 0 or stop == n_items):
            print(f"  block {bi + 1}/{n_blocks}  items {start}-{stop}")

    S = sp.csr_matrix(
        (np.concatenate(vals_out), (np.concatenate(rows_out), np.concatenate(cols_out))),
        shape=(n_items, n_items),
        dtype="float32",
    )
    if verbose:
        print(f"  similarity matrix: {S.shape}, {S.nnz:,} nonzeros")
    return S


def save_similarity(
    S: sp.csr_matrix, ml: MovieLens, z_mean: np.ndarray | None = None
) -> None:
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    sp.save_npz(SIM_NPZ, S)
    if z_mean is None:
        z_mean = item_z_means(ml)
    np.savez_compressed(
        ITEM_STATS_NPZ,
        n_ratings=ml.items["n_ratings"].to_numpy(dtype="int32"),
        mean_rating=ml.items["mean_rating"].to_numpy(dtype="float32"),
        z_mean=z_mean.astype("float32"),
    )


# ---------------------------------------------------------------- scoring


@dataclass
class Scored:
    item_idx: np.ndarray
    pred_z: np.ndarray  # honest absolute prediction, in the user's z units
    support: np.ndarray
    n_neighbours: np.ndarray
    rank_score: np.ndarray
    pred_d: np.ndarray | None = None  # predicted disagreement with the crowd

    def to_frame(self, ml: MovieLens) -> pd.DataFrame:
        df = pd.DataFrame(
            {
                "item_idx": self.item_idx,
                "pred_z": self.pred_z,
                "support": self.support,
                "n_neighbours": self.n_neighbours,
                "rank_score": self.rank_score,
                "pred_d": (
                    self.pred_d if self.pred_d is not None else np.zeros_like(self.pred_z)
                ),
            }
        )
        cols = ml.items[["item_idx", "clean_title", "year", "n_ratings", "mean_rating", "tmdbId", "genres"]]
        return df.merge(cols, on="item_idx", how="left")


class ItemItemCF:
    """Loaded similarity matrix plus the scoring logic."""

    def __init__(
        self,
        S: sp.csr_matrix,
        n_ratings: np.ndarray,
        config: CFConfig = CF,
        z_mean: np.ndarray | None = None,
    ):
        self.S = S
        self.n_ratings = n_ratings
        self.config = config
        # The consensus baseline. Zeros is the honest fallback for a matrix built
        # before this existed: it degrades to the old behaviour rather than
        # inventing a consensus, and the caller sees it in the eval numbers.
        self.z_mean = (
            np.zeros(S.shape[0], dtype="float32") if z_mean is None else z_mean.astype("float32")
        )
        self._median_n = float(np.median(n_ratings[n_ratings > 0]))

    @classmethod
    def load(cls, config: CFConfig = CF) -> ItemItemCF:
        if not SIM_NPZ.exists():
            raise FileNotFoundError(
                f"{SIM_NPZ} missing. Run: python -m scripts.build_cf"
            )
        S = sp.load_npz(SIM_NPZ).tocsr()
        stats = np.load(ITEM_STATS_NPZ)
        z_mean = stats["z_mean"] if "z_mean" in stats.files else None
        if z_mean is None:
            print(
                "  ! item_stats.npz predates consensus decomposition and has no "
                "z_mean; falling back to zeros (i.e. the old scorer). "
                "Re-run scripts.build_cf to fix."
            )
        return cls(S, stats["n_ratings"], config, z_mean)

    def input_values(self, rated_items: np.ndarray, z: np.ndarray) -> np.ndarray:
        """What actually goes into the weighted average: z, or z minus consensus.

        Shared by score() and explain() so that the explanation is always in
        terms of the quantity the ranking was actually computed from.
        """
        if not self.config.use_residual_input:
            return np.asarray(z, dtype="float64")
        return np.asarray(z, dtype="float64") - self.z_mean[rated_items]

    def score(
        self,
        rated_items: np.ndarray,
        z: np.ndarray,
        weights: np.ndarray | None = None,
        exclude: np.ndarray | None = None,
    ) -> Scored:
        """Score every film against one person's ratings.

        rated_items / z / weights are parallel arrays over the films this person
        has rated and that we managed to match. `exclude` is everything to filter
        out of the results -- which must be the union of everything they have
        WATCHED, not merely everything they rated.
        """
        if weights is None:
            weights = np.ones(len(rated_items), dtype="float32")

        rated_items = np.asarray(rated_items, dtype="int64")
        cfg = self.config

        # z, or z with the crowd subtracted out -- see CFConfig.use_residual_input
        # for why the second one is what makes this beat the consensus baseline.
        values = self.input_values(rated_items, z)

        n_items = self.S.shape[0]
        num = np.zeros(n_items, dtype="float64")
        den = np.zeros(n_items, dtype="float64")
        cnt = np.zeros(n_items, dtype="int32")

        # Walk the rows of S belonging to films they rated, accumulating the
        # weighted average described in the module docstring.
        indptr, indices, data = self.S.indptr, self.S.indices, self.S.data
        for item, vi, wi in zip(rated_items, values, weights):
            lo, hi = indptr[item], indptr[item + 1]
            cols = indices[lo:hi]
            sims = data[lo:hi]
            num[cols] += sims * (vi * wi)
            den[cols] += np.abs(sims) * wi
            cnt[cols] += 1

        pred = np.divide(num, den, out=np.zeros_like(num), where=den > 1e-9)

        # --- support shrinkage ---
        #
        # Without this every prediction came back 4.6-4.96 stars: a weighted
        # average over three neighbours just hands back those neighbours' values,
        # with no memory of how little was behind them. Pulling thin predictions
        # toward 0 pulls the star estimate toward the user's own mean, which is
        # the correct thing to say when we know almost nothing.
        if cfg.pred_shrinkage_k > 0:
            pred = pred * (den / (den + cfg.pred_shrinkage_k))

        # How much the personal term is trusted in the RANKING, by evidence.
        # See CFConfig.rank_shrinkage_k: this is what stops a film with one weak
        # neighbour from taking the top slot on a prediction built out of
        # nothing, and it is what makes a low min_support safe to run.
        if cfg.rank_shrinkage_k > 0:
            trust = den / (den + cfg.rank_shrinkage_k)
        else:
            trust = np.ones_like(den)

        if cfg.use_residual_input:
            # pred is predicted DISAGREEMENT. The honest absolute prediction adds
            # the crowd's own standing back in; the ranking adds back only as
            # much of it as beta asks for, and trusts the personal part only as
            # far as the evidence goes.
            pred_d = pred
            pred_z = self.z_mean.astype("float64") + pred_d
            base = cfg.consensus_beta * self.z_mean.astype("float64") + pred_d * trust
        else:
            pred_d = None
            pred_z = pred
            base = pred * trust

        # Novelty penalty, in z-units per decade of popularity.
        pop = np.maximum(self.n_ratings.astype("float64"), 1.0)
        penalty = cfg.popularity_alpha * np.log10(pop / self._median_n)
        rank_score = base - penalty

        # Anything we have no business making a claim about is pushed to -inf
        # rather than silently ranked.
        weak = den < cfg.min_support
        rank_score[weak] = -np.inf

        mask = np.ones(n_items, dtype=bool)
        mask[rated_items] = False
        if exclude is not None and len(exclude):
            mask[exclude] = False
        keep = np.flatnonzero(mask)

        return Scored(
            item_idx=keep,
            pred_z=pred_z[keep],
            support=den[keep],
            n_neighbours=cnt[keep],
            rank_score=rank_score[keep],
            pred_d=None if pred_d is None else pred_d[keep],
        )

    def explain(self, candidate: int, rated_items: np.ndarray, z: np.ndarray, top: int = 5):
        """Which of their films drove this recommendation, and in which direction.

        This is the interpretability payoff of item-item CF, and it is what the
        LLM layer will be handed instead of being asked to invent a reason.
        """
        # Read the SAME direction the scorer reads.
        #
        # S is truncated to the top-K neighbours of each row, which makes it
        # asymmetric: j can be in i's row while i is absent from j's. score()
        # walks the rows of the films you RATED and accumulates into candidates,
        # so a contribution exists when the rated film's row contains the
        # candidate. Reading the candidate's row instead -- which this used to do
        # -- answers a different question, and silently disagreed with the number
        # it was supposed to explain: "Broken (2014)" scored from 2 neighbours and
        # explained itself with 0, printing a recommendation with no reason.
        rated_items = np.asarray(rated_items, dtype="int64")
        # In residual mode the driver is "you disagreed with the crowd about X",
        # not "you liked X", so explain in the units the ranking used.
        values = self.input_values(rated_items, z)
        indptr, indices, data = self.S.indptr, self.S.indices, self.S.data

        contribs: list[tuple[int, float]] = []
        for i, vi in zip(rated_items, values):
            lo, hi = indptr[i], indptr[i + 1]
            hit = np.flatnonzero(indices[lo:hi] == candidate)
            if len(hit):
                contribs.append((int(i), float(data[lo + hit[0]] * vi)))
        contribs.sort(key=lambda t: -abs(t[1]))
        return contribs[:top]
