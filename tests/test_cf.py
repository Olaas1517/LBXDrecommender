"""The scoring math -- including the property the whole rework turns on."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from lbxd.cf import ItemItemCF, build_similarity, item_z_means, user_zscores
from lbxd.config import CF
from tests.conftest import CONSENSUS_ITEM, N_ARTHOUSE, N_ITEMS


@pytest.fixture
def engine(mini_ml):
    cfg = replace(CF, shrinkage_lambda=5.0, top_k_neighbours=12, min_support=0.0)
    S = build_similarity(mini_ml, cfg, block_size=8, verbose=False)
    z_mean = item_z_means(mini_ml, config=cfg)
    return ItemItemCF(S, mini_ml.items["n_ratings"].to_numpy(), cfg, z_mean), mini_ml


def test_user_zscores_are_centred_per_user(mini_ml):
    z = user_zscores(mini_ml)
    for u in np.unique(mini_ml.user_idx)[:5]:
        assert abs(z[mini_ml.user_idx == u].mean()) < 1e-5


def test_similar_films_are_the_ones_in_the_same_cluster(engine):
    cf, ml = engine
    row = cf.S[0].toarray().ravel()
    within = row[1:N_ARTHOUSE].mean()
    across = row[N_ARTHOUSE:N_ARTHOUSE + 6].mean()
    assert within > across, "arthouse films should resemble each other, not blockbusters"


def test_a_film_is_not_its_own_neighbour(engine):
    cf, _ = engine
    assert cf.S[3, 3] == 0.0


def test_shrinkage_discards_thin_evidence():
    """Two films co-rated by three people must not look like a perfect match.

    This is the single most important line in cf.py, so it gets an exact
    arithmetic check rather than a vibe check.

    The three raters must also rate a THIRD film differently, or they have zero
    rating variance, their z-scores are all 0, and the fixture proves nothing --
    which is exactly the trap this test fell into first time round.
    """
    import pandas as pd
    from lbxd.movielens import MovieLens

    # Users 0-2 adore items 0 and 1 identically, and dislike item 2.
    u = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2] + list(range(3, 60)), dtype="int32")
    i = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2] + [2] * 57, dtype="int32")
    r = np.array([5.0, 5.0, 1.0] * 3 + [3.0] * 57, dtype="float32")
    items = pd.DataFrame(
        {
            "item_idx": [0, 1, 2], "movieId": [1, 2, 3],
            "clean_title": ["A", "B", "C"], "title": ["A", "B", "C"],
            "year": pd.array([2000, 2001, 2002], dtype="Int64"),
            "genres": ["Drama"] * 3,
            "imdbId": pd.array(["1", "2", "3"], dtype="string"),
            "tmdbId": pd.array(["1", "2", "3"], dtype="string"),
            "n_ratings": np.array([3, 3, 60], dtype="int32"),
            "mean_rating": np.array([5.0, 5.0, 2.9], dtype="float32"),
        }
    )
    ml = MovieLens(user_idx=u, item_idx=i, rating=r, items=items)

    unshrunk = build_similarity(ml, replace(CF, shrinkage_lambda=0.0), block_size=4, verbose=False)
    shrunk = build_similarity(ml, replace(CF, shrinkage_lambda=60.0), block_size=4, verbose=False)

    assert np.isfinite(unshrunk.data).all(), "lambda=0 must not produce NaN similarities"
    assert unshrunk[0, 1] == pytest.approx(1.0, abs=1e-5), (
        "identically-rated films should look identical before shrinkage"
    )
    # n_ij = 3, lambda = 60  ->  exactly 3/63 of the raw value.
    assert shrunk[0, 1] == pytest.approx(3 / 63, rel=1e-4)
    assert shrunk[0, 1] < 0.05, "3 co-raters is not evidence of anything"


def test_scoring_ranks_your_own_cluster_first(engine):
    cf, ml = engine
    # Someone who loves arthouse: rate items 0-2 high, 6-8 low.
    rated = np.array([0, 1, 2, 6, 7, 8])
    z = np.array([1.2, 1.1, 1.3, -1.2, -1.1, -1.3])
    scored = cf.score(rated, z)
    pos = {int(it): p for p, it in enumerate(scored.item_idx)}

    unseen_arthouse = [pos[i] for i in range(3, N_ARTHOUSE)]
    unseen_blockbuster = [pos[i] for i in range(9, 12)]
    assert scored.pred_z[unseen_arthouse].mean() > scored.pred_z[unseen_blockbuster].mean()


def test_rated_films_are_excluded_from_results(engine):
    cf, _ = engine
    rated = np.array([0, 1, 2])
    scored = cf.score(rated, np.array([1.0, 1.0, 1.0]))
    assert not set(rated) & set(scored.item_idx.tolist())


def test_exclude_list_is_honoured(engine):
    cf, _ = engine
    scored = cf.score(np.array([0]), np.array([1.0]), exclude=np.array([5, 6]))
    assert not {5, 6} & set(scored.item_idx.tolist())


# --------------------------------------------------------------------------
# The property the whole consensus rework depends on.
# --------------------------------------------------------------------------


def test_an_average_user_reduces_to_the_consensus_ranking(engine):
    """A user who agrees with the crowd about everything must be told what the
    crowd likes -- and nothing more.

    This is the load-bearing property of the residual decomposition. If someone
    has zero disagreement with consensus, there is no personal signal to act on,
    so the honest answer IS the consensus ranking. A scorer that returned
    something else for such a user would be inventing personalisation out of
    noise.
    """
    cf, ml = engine
    cfg = replace(cf.config, use_residual_input=True, consensus_beta=1.0,
                  popularity_alpha=0.0, pred_shrinkage_k=0.0, min_support=0.0)
    cf.config = cfg

    rated = np.arange(N_ITEMS - 4)
    z = cf.z_mean[rated].astype("float64")  # exactly the crowd's view
    scored = cf.score(rated, z)

    # pred_d must be ~0 everywhere, so pred_z is the consensus itself.
    assert np.abs(scored.pred_d).max() < 1e-6
    np.testing.assert_allclose(scored.pred_z, cf.z_mean[scored.item_idx], atol=1e-6)


def test_consensus_cannot_free_ride_when_beta_is_zero(engine):
    """A film whose only claim is that everyone loves it must not ride to the
    top of a maximally-personal ranking.

    Constructed rather than sampled: one candidate is handed an enormous
    consensus score and no personal signal at all. At beta=1 it should dominate
    (that is what beta=1 means -- rank by predicted absolute enjoyment). At
    beta=0 it must not, because "the crowd adores it" is precisely the claim
    beta=0 says to ignore.

    NOTE this is a test of the DIAL, not evidence that residual scoring makes
    better recommendations. That claim is empirical, it is about real rating
    data, and a toy fixture cannot settle it -- see scripts.tune for the
    measurement that can.
    """
    cf, ml = engine
    cf.config = replace(cf.config, use_residual_input=True, popularity_alpha=0.0,
                        pred_shrinkage_k=0.0, min_support=0.0)

    darling = 11  # a blockbuster our arthouse user has no affinity for
    cf.z_mean = cf.z_mean.copy()
    cf.z_mean[darling] = 5.0  # adored by everyone, on any scale

    rated = np.array([0, 1, 2, 6, 7, 8])
    z = np.array([1.2, 1.1, 1.3, -1.2, -1.1, -1.3])

    def rank_of_darling(beta):
        cf.config = replace(cf.config, consensus_beta=beta)
        s = cf.score(rated, z)
        order = np.argsort(-s.rank_score, kind="stable")
        return list(s.item_idx[order]).index(darling)

    assert rank_of_darling(1.0) == 0, "beta=1 should rank the adored film first"
    assert rank_of_darling(0.0) > 0, "beta=0 must ignore pure consensus"


def test_residual_input_removes_the_crowd_from_the_signal(engine):
    """The mechanical claim: what enters the average is z minus the crowd."""
    cf, _ = engine
    rated = np.array([0, 5, 9])
    z = np.array([1.0, -0.5, 0.25])

    cf.config = replace(cf.config, use_residual_input=True)
    np.testing.assert_allclose(cf.input_values(rated, z), z - cf.z_mean[rated], atol=1e-6)

    cf.config = replace(cf.config, use_residual_input=False)
    np.testing.assert_allclose(cf.input_values(rated, z), z, atol=1e-6)


def test_support_shrinkage_pulls_thin_predictions_toward_the_user_mean(engine):
    """The fix for 'every prediction comes back 4.6-4.96 stars'."""
    cf, _ = engine
    rated = np.array([0, 1])
    z = np.array([2.0, 2.0])

    cf.config = replace(cf.config, pred_shrinkage_k=0.0, use_residual_input=False,
                        min_support=0.0, popularity_alpha=0.0)
    unshrunk = cf.score(rated, z)
    cf.config = replace(cf.config, pred_shrinkage_k=3.0)
    shrunk = cf.score(rated, z)

    assert np.abs(shrunk.pred_z).max() < np.abs(unshrunk.pred_z).max()
    # Shrinkage may only pull toward zero, never past it or away from it.
    assert np.all(np.abs(shrunk.pred_z) <= np.abs(unshrunk.pred_z) + 1e-9)
    assert np.all(np.sign(shrunk.pred_z) * np.sign(unshrunk.pred_z) >= 0)


def test_beta_dials_consensus_in_and_out(engine):
    """beta must move the ranking monotonically toward the consensus order."""
    cf, ml = engine
    rated = np.array([0, 1, 2])
    z = np.array([1.5, 1.4, 1.6])
    cf.config = replace(cf.config, use_residual_input=True, popularity_alpha=0.0,
                        pred_shrinkage_k=0.0, min_support=0.0)

    def corr_with_consensus(beta):
        cf.config = replace(cf.config, consensus_beta=beta)
        s = cf.score(rated, z)
        return np.corrcoef(s.rank_score, cf.z_mean[s.item_idx])[0, 1]

    assert corr_with_consensus(1.0) > corr_with_consensus(0.0)


def test_legacy_mode_is_exactly_the_old_scorer(engine):
    """use_residual_input=False must reproduce pre-rework behaviour, so the
    sweep's old-vs-new comparison is a real comparison."""
    cf, _ = engine
    rated = np.array([0, 1, 6])
    z = np.array([1.0, 0.5, -1.0])
    cf.config = replace(cf.config, use_residual_input=False, pred_shrinkage_k=0.0,
                        popularity_alpha=0.0, min_support=0.0)
    s = cf.score(rated, z)

    # Hand-compute the old weighted average for one candidate.
    j = int(s.item_idx[0])
    num = den = 0.0
    for it, zi in zip(rated, z):
        sim = cf.S[it, j]
        num += sim * zi
        den += abs(sim)
    expected = num / den if den > 1e-9 else 0.0
    assert s.pred_z[0] == pytest.approx(expected, rel=1e-5, abs=1e-8)
    assert s.pred_d is None


def test_low_support_is_refused_not_guessed(engine):
    cf, _ = engine
    cf.config = replace(cf.config, min_support=1e9)
    s = cf.score(np.array([0]), np.array([1.0]))
    assert np.all(~np.isfinite(s.rank_score)), "unsupported films must be unrankable"


def test_item_centring_removes_each_films_own_mean(mini_ml):
    """Pearson, not adjusted cosine: after centring, every film's observed
    ratings must average to zero, or the similarity is not a correlation."""
    import scipy.sparse as sp
    from lbxd.cf import user_zscores

    cfg = replace(CF, item_centered=True, shrinkage_lambda=5.0, top_k_neighbours=12)
    z = user_zscores(mini_ml)
    Z = sp.csr_matrix(
        (z, (mini_ml.item_idx, mini_ml.user_idx)),
        shape=(mini_ml.n_items, int(mini_ml.user_idx.max()) + 1), dtype="float32",
    )
    counts = np.diff(Z.indptr).astype("float64")
    row_mean = np.asarray(Z.sum(axis=1)).ravel() / counts
    centred = Z.data - np.repeat(row_mean, np.diff(Z.indptr))

    for i in range(mini_ml.n_items):
        lo, hi = Z.indptr[i], Z.indptr[i + 1]
        assert abs(centred[lo:hi].mean()) < 1e-5, f"film {i} not centred"

    # And the build must still produce a finite, self-zeroed matrix.
    S = build_similarity(mini_ml, cfg, block_size=8, verbose=False)
    assert np.isfinite(S.data).all()
    assert S[3, 3] == 0.0


def test_item_centring_changes_what_similar_means(mini_ml):
    """Two films that are merely both well-liked should look LESS alike once
    each film's own mean is removed -- that shared positive offset is
    co-popularity, not co-taste."""
    base = replace(CF, item_centered=False, shrinkage_lambda=5.0, top_k_neighbours=12)
    cent = replace(base, item_centered=True)
    S0 = build_similarity(mini_ml, base, block_size=8, verbose=False)
    S1 = build_similarity(mini_ml, cent, block_size=8, verbose=False)
    assert not np.allclose(S0.toarray(), S1.toarray()), "centring must change the matrix"


def test_thin_evidence_cannot_win_the_ranking(engine):
    """The regression this exists to stop: a film with one weak neighbour was
    handed pred_d = +1.96z and took the top slot with "support 0.0".

    rank_shrinkage_k discounts the personal term by the evidence behind it, so a
    near-zero-support film falls back to the crowd's view instead of winning.
    """
    cf, _ = engine
    cf.config = replace(cf.config, use_residual_input=True, consensus_beta=1.0,
                        popularity_alpha=0.0, min_support=0.0, rank_shrinkage_k=1.0)
    rated = np.array([0, 1, 2])
    z = np.array([1.5, 1.4, 1.6])
    s = cf.score(rated, z)

    thin = s.support < 0.05
    if thin.any():
        # Where there is no evidence, rank_score must be ~the consensus term.
        np.testing.assert_allclose(
            s.rank_score[thin], cf.z_mean[s.item_idx][thin], atol=0.1
        )

    # And with the discount off, the thin films' personal term is NOT damped.
    cf.config = replace(cf.config, rank_shrinkage_k=0.0)
    undamped = cf.score(rated, z)
    if thin.any():
        assert np.abs(undamped.rank_score[thin] - cf.z_mean[undamped.item_idx][thin]).max() >= \
               np.abs(s.rank_score[thin] - cf.z_mean[s.item_idx][thin]).max()


def test_rank_shrinkage_leaves_the_star_prediction_alone(engine):
    """pred_z must stay an honest statement about the evidence -- only the
    RANKING is discounted, or the falsifiable claim stops being falsifiable."""
    cf, _ = engine
    rated = np.array([0, 1, 2])
    z = np.array([1.5, 1.4, 1.6])
    cf.config = replace(cf.config, use_residual_input=True, min_support=0.0,
                        popularity_alpha=0.0, rank_shrinkage_k=0.0)
    a = cf.score(rated, z)
    cf.config = replace(cf.config, rank_shrinkage_k=5.0)
    b = cf.score(rated, z)
    np.testing.assert_allclose(a.pred_z, b.pred_z, atol=1e-9)
    assert not np.allclose(a.rank_score, b.rank_score)
