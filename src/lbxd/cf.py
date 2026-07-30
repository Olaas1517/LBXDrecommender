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


# ---------------------------------------------------------------- precompute


def build_similarity(
    ml: MovieLens,
    config: CFConfig = CF,
    block_size: int = 512,
    verbose: bool = True,
) -> sp.csr_matrix:
    """Precompute the shrunk, top-K-truncated item-item similarity matrix."""
    z = user_zscores(ml)
    n_items = ml.n_items
    n_users = int(ml.user_idx.max()) + 1

    # Items x users, values = z-scores. CSR so that row blocks are cheap to slice.
    Z = sp.csr_matrix(
        (z, (ml.item_idx, ml.user_idx)), shape=(n_items, n_users), dtype="float32"
    )
    # Binary twin, for counting co-raters.
    B = sp.csr_matrix(
        (np.ones(len(z), dtype="float32"), (ml.item_idx, ml.user_idx)),
        shape=(n_items, n_users),
        dtype="float32",
    )

    # L2-normalise each item row so that a dot product IS the cosine.
    norms = np.sqrt(Z.multiply(Z).sum(axis=1)).A.ravel()
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
        np.multiply(sim, cnt / (cnt + config.shrinkage_lambda), out=sim)

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
            keep = v != 0
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


def save_similarity(S: sp.csr_matrix, ml: MovieLens) -> None:
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    sp.save_npz(SIM_NPZ, S)
    np.savez_compressed(
        ITEM_STATS_NPZ,
        n_ratings=ml.items["n_ratings"].to_numpy(dtype="int32"),
        mean_rating=ml.items["mean_rating"].to_numpy(dtype="float32"),
    )


# ---------------------------------------------------------------- scoring


@dataclass
class Scored:
    item_idx: np.ndarray
    pred_z: np.ndarray
    support: np.ndarray
    n_neighbours: np.ndarray
    rank_score: np.ndarray

    def to_frame(self, ml: MovieLens) -> pd.DataFrame:
        df = pd.DataFrame(
            {
                "item_idx": self.item_idx,
                "pred_z": self.pred_z,
                "support": self.support,
                "n_neighbours": self.n_neighbours,
                "rank_score": self.rank_score,
            }
        )
        cols = ml.items[["item_idx", "clean_title", "year", "n_ratings", "mean_rating", "tmdbId", "genres"]]
        return df.merge(cols, on="item_idx", how="left")


class ItemItemCF:
    """Loaded similarity matrix plus the scoring logic."""

    def __init__(self, S: sp.csr_matrix, n_ratings: np.ndarray, config: CFConfig = CF):
        self.S = S
        self.n_ratings = n_ratings
        self.config = config
        self._median_n = float(np.median(n_ratings[n_ratings > 0]))

    @classmethod
    def load(cls, config: CFConfig = CF) -> ItemItemCF:
        if not SIM_NPZ.exists():
            raise FileNotFoundError(
                f"{SIM_NPZ} missing. Run: py -m scripts.build_cf"
            )
        S = sp.load_npz(SIM_NPZ).tocsr()
        stats = np.load(ITEM_STATS_NPZ)
        return cls(S, stats["n_ratings"], config)

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

        n_items = self.S.shape[0]
        num = np.zeros(n_items, dtype="float64")
        den = np.zeros(n_items, dtype="float64")
        cnt = np.zeros(n_items, dtype="int32")

        # Walk the rows of S belonging to films they rated, accumulating the
        # weighted average described in the module docstring.
        indptr, indices, data = self.S.indptr, self.S.indices, self.S.data
        for item, zi, wi in zip(rated_items, z, weights):
            lo, hi = indptr[item], indptr[item + 1]
            cols = indices[lo:hi]
            sims = data[lo:hi]
            num[cols] += sims * (zi * wi)
            den[cols] += np.abs(sims) * wi
            cnt[cols] += 1

        pred_z = np.divide(num, den, out=np.zeros_like(num), where=den > 1e-9)

        # Novelty penalty, in z-units per decade of popularity.
        pop = np.maximum(self.n_ratings.astype("float64"), 1.0)
        penalty = self.config.popularity_alpha * np.log10(pop / self._median_n)
        rank_score = pred_z - penalty

        # Anything we have no business making a claim about is pushed to -inf
        # rather than silently ranked.
        weak = den < self.config.min_support
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
        )

    def explain(self, candidate: int, rated_items: np.ndarray, z: np.ndarray, top: int = 5):
        """Which of their films drove this recommendation, and in which direction.

        This is the interpretability payoff of item-item CF, and it is what the
        LLM layer will be handed instead of being asked to invent a reason.
        """
        row = self.S[candidate]
        sim_by_item = dict(zip(row.indices, row.data))
        contribs = [
            (int(i), float(sim_by_item.get(int(i), 0.0) * zi))
            for i, zi in zip(rated_items, z)
            if int(i) in sim_by_item
        ]
        contribs.sort(key=lambda t: -abs(t[1]))
        return contribs[:top]
