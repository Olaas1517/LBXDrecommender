"""Offline evaluation. Build this before trusting a single recommendation.

The method: hide a random slice of the films you have already rated, run
retrieval as though they were unseen, and check whether the ones you actually
liked come back near the top. Your own ratings are the only ground truth
available, which is why your export is a test set rather than training data.

=============================================================================
WHY THE BASELINES MATTER MORE THAN THE SCORES
=============================================================================

A recall@50 of 0.08 means nothing in isolation. Ranking ~23,000 films with only
~90 held-out relevant items produces small numbers no matter how good the system
is. What is meaningful is the comparison against baselines that require no
personalisation at all:

  random        the floor. If you cannot beat this, something is broken.
  by consensus  rank by the film's own average rating -- i.e. "just recommend
                critically acclaimed films to everybody". This is the baseline
                that actually matters, and it is the one most recommenders
                secretly fail to beat. For a contrarian rater it should be
                beatable by a wide margin, because consensus is precisely what
                they disagree with.
  by popularity rank by rating count -- "recommend what everyone has seen".

If personalised scoring cannot beat "by consensus", then the personalisation is
decorative and you have built an expensive way to suggest The Godfather.

=============================================================================
THE HEADLINE METRIC
=============================================================================

recall@k is noisy at this scale, so the primary number here is the median
PERCENTILE RANK of held-out films you liked. If a film you rated 4.5 lands at
percentile 0.98, the system put it in the top 2% of ~23,000 candidates -- that is
legible, and it stays stable with small holdouts. recall@k and NDCG@k are reported
too, because they are what the literature uses.

Everything is also broken down by popularity quintile, because accuracy on
crowd-pleasers is easy and uninteresting. What matters is whether the system is
still right about films outside the top quintile -- accuracy conditional on
non-obviousness.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .cf import ItemItemCF
from .config import EVAL, EvalConfig
from .movielens import MovieLens


@dataclass
class EvalReport:
    n_train: int
    n_test: int
    n_relevant: int
    n_relevant_scoreable: int
    metrics: dict[str, float] = field(default_factory=dict)
    baselines: dict[str, dict[str, float]] = field(default_factory=dict)
    by_stratum: pd.DataFrame | None = None
    rating_error: dict[str, float] = field(default_factory=dict)
    # method name -> {"Q1": median percentile, ...}, for CF and every baseline
    stratum_by_method: dict[str, dict[str, float]] = field(default_factory=dict)

    def summary(self) -> str:
        L = [
            f"train films        : {self.n_train}",
            f"held-out films     : {self.n_test}",
            f"  of which liked   : {self.n_relevant}  (z >= threshold)",
            f"  scoreable at all : {self.n_relevant_scoreable}"
            f"  ({self.n_relevant_scoreable / max(self.n_relevant, 1):.0%}"
            " -- the rest have too little similarity support to rank)",
            "",
            "HEADLINE  median percentile rank of held-out liked films",
            "          (1.00 = ranked top of ~23,000 candidates; 0.50 = coin flip)",
        ]
        rows = [("item-item CF", self.metrics.get("median_percentile", float("nan")))]
        for name, m in self.baselines.items():
            rows.append((name, m.get("median_percentile", float("nan"))))
        for name, val in rows:
            bar = "#" * int(max(val, 0) * 40) if val == val else ""
            L.append(f"    {name:<16} {val:.3f}  {bar}")

        L.append("")
        L.append("recall@k / NDCG@k")
        header = "    {:<16}".format("") + "".join(f"  R@{k:<5}" for k in EVAL.ks) + "".join(
            f"  N@{k:<5}" for k in EVAL.ks
        )
        L.append(header)
        def fmt(m: dict[str, float]) -> str:
            return "".join(f"  {m.get(f'recall@{k}', 0):.3f}" for k in EVAL.ks) + "".join(
                f"  {m.get(f'ndcg@{k}', 0):.3f}" for k in EVAL.ks
            )
        L.append(f"    {'item-item CF':<16}" + fmt(self.metrics))
        for name, m in self.baselines.items():
            L.append(f"    {name:<16}" + fmt(m))

        if self.rating_error:
            L.append("")
            L.append("predicted-star accuracy on held-out films (the falsifiable claim)")
            L.append(f"    MAE  {self.rating_error['mae']:.3f} stars")
            L.append(f"    RMSE {self.rating_error['rmse']:.3f} stars")
            L.append(f"    n    {int(self.rating_error['n'])}")
            L.append(f"    (baseline: always predict your mean -> MAE {self.rating_error['mae_trivial']:.3f})")

        if self.stratum_by_method:
            L.append("")
            L.append("THE COMPARISON THAT MATTERS")
            L.append("median percentile of held-out liked films, BY POPULARITY QUINTILE")
            L.append("(Q1 = least widely seen. Aggregate recall is dominated by popular")
            L.append(" films, so this is where personalisation has to earn its keep.)")
            qs = sorted({q for m in self.stratum_by_method.values() for q in m})
            counts = {}
            if self.by_stratum is not None and not self.by_stratum.empty:
                counts = dict(zip(self.by_stratum["quintile"], self.by_stratum["n"]))
            L.append("    {:<16}".format("") + "".join(f"{q:>9}" for q in qs))
            L.append("    {:<16}".format("n =") + "".join(f"{int(counts.get(q, 0)):>9}" for q in qs))
            for name, m in self.stratum_by_method.items():
                L.append(
                    f"    {name:<16}" + "".join(
                        (f"{m[q]:>9.3f}" if q in m else f"{'-':>9}") for q in qs
                    )
                )
        return "\n".join(L)


def _ndcg_at_k(ranked_relevant: np.ndarray, n_relevant: int, k: int) -> float:
    """ranked_relevant: boolean array over the top-k, in rank order."""
    gains = ranked_relevant[:k].astype("float64")
    discounts = 1.0 / np.log2(np.arange(2, len(gains) + 2))
    dcg = float((gains * discounts).sum())
    ideal_n = min(n_relevant, k)
    idcg = float((np.ones(ideal_n) / np.log2(np.arange(2, ideal_n + 2))).sum())
    return dcg / idcg if idcg > 0 else 0.0


def _metrics_for_ranking(
    order: np.ndarray, relevant_positions: set[int], n_relevant: int, ks
) -> dict[str, float]:
    """order: candidate indices best-first. relevant_positions: indices that count."""
    is_rel = np.fromiter((int(i) in relevant_positions for i in order), dtype=bool, count=len(order))
    out: dict[str, float] = {}
    for k in ks:
        out[f"recall@{k}"] = float(is_rel[:k].sum()) / max(n_relevant, 1)
        out[f"ndcg@{k}"] = _ndcg_at_k(is_rel, n_relevant, k)

    n = len(order)
    ranks = np.flatnonzero(is_rel)
    if len(ranks):
        # percentile: 1.0 means ranked first of n
        pct = 1.0 - ranks / max(n - 1, 1)
        out["median_percentile"] = float(np.median(pct))
        out["best_percentile"] = float(pct.max())
    else:
        out["median_percentile"] = 0.0
        out["best_percentile"] = 0.0
    return out


def evaluate(
    matched: pd.DataFrame,
    cf: ItemItemCF,
    ml: MovieLens,
    profile_mean: float,
    profile_sigma: float,
    exclude_item_idx: np.ndarray | None = None,
    config: EvalConfig = EVAL,
    n_repeats: int = 10,
) -> EvalReport:
    """Repeated random holdout, pooled.

    WHY REPEATS: a single 20% split of ~440 films yields only ~30 "liked"
    held-out items. Split those across five popularity quintiles and you get
    cells of n=1, whose median is meaningless -- you end up tuning on noise.
    Pooling ten independent splits gives ~300 observations and makes the
    stratified numbers, which are the ones that actually matter, readable.
    """
    reports = [
        _evaluate_once(
            matched, cf, ml, profile_mean, profile_sigma, exclude_item_idx, config, seed_offset=i
        )
        for i in range(n_repeats)
    ]

    def avg(get) -> dict[str, float]:
        keys = set().union(*(set(get(r).keys()) for r in reports))
        return {k: float(np.mean([get(r).get(k, 0.0) for r in reports])) for k in keys}

    pooled_strata = pd.concat(
        [r.by_stratum for r in reports if r.by_stratum is not None], ignore_index=True
    ) if any(r.by_stratum is not None for r in reports) else None
    if pooled_strata is not None and not pooled_strata.empty:
        pooled_strata = (
            pooled_strata.groupby("quintile")
            .apply(
                lambda g: pd.Series(
                    {
                        "n": int(g["n"].sum()),
                        "median_percentile": float(
                            np.average(g["median_percentile"], weights=g["n"])
                        ),
                    }
                ),
                include_groups=False,
            )
            .reset_index()
        )

    base_names = reports[0].baselines.keys()
    return EvalReport(
        n_train=reports[0].n_train,
        n_test=reports[0].n_test,
        n_relevant=int(np.mean([r.n_relevant for r in reports])),
        n_relevant_scoreable=int(np.mean([r.n_relevant_scoreable for r in reports])),
        metrics=avg(lambda r: r.metrics),
        baselines={nm: avg(lambda r, nm=nm: r.baselines[nm]) for nm in base_names},
        by_stratum=pooled_strata,
        rating_error=avg(lambda r: r.rating_error),
        stratum_by_method={
            nm: avg(lambda r, nm=nm: r.stratum_by_method.get(nm, {}))
            for nm in reports[0].stratum_by_method
        },
    )


def _evaluate_once(
    matched: pd.DataFrame,
    cf: ItemItemCF,
    ml: MovieLens,
    profile_mean: float,
    profile_sigma: float,
    exclude_item_idx: np.ndarray | None = None,
    config: EvalConfig = EVAL,
    seed_offset: int = 0,
) -> EvalReport:
    """matched needs columns: item_idx, z, weight, rating."""
    rng = np.random.default_rng(config.seed + seed_offset)

    df = matched.dropna(subset=["item_idx"]).copy()
    df["item_idx"] = df["item_idx"].astype(int)
    df = df.drop_duplicates(subset=["item_idx"])

    n = len(df)
    n_test = max(int(round(n * config.holdout_frac)), 1)
    perm = rng.permutation(n)
    test_df = df.iloc[perm[:n_test]]
    train_df = df.iloc[perm[n_test:]]

    train_items = train_df["item_idx"].to_numpy()
    train_z = train_df["z"].to_numpy(dtype="float32")
    train_w = train_df["weight"].to_numpy(dtype="float32") if "weight" in train_df else None

    # Exclude only the TRAIN films from the candidate pool. The held-out films
    # must remain rankable -- they are the answers. Films the user watched but
    # never rated are also excluded, since a real run would never surface them.
    exclude = exclude_item_idx
    if exclude is not None:
        exclude = np.setdiff1d(np.asarray(exclude, dtype=int), test_df["item_idx"].to_numpy())

    scored = cf.score(train_items, train_z, train_w, exclude=exclude)

    pos_of_item = {int(it): i for i, it in enumerate(scored.item_idx)}
    relevant = test_df[test_df["z"] >= config.relevant_z_threshold]
    rel_positions = {pos_of_item[i] for i in relevant["item_idx"] if i in pos_of_item}
    n_relevant = len(rel_positions)

    scoreable = sum(1 for p in rel_positions if np.isfinite(scored.rank_score[p]))

    ks = config.ks
    order = np.argsort(-scored.rank_score, kind="stable")
    metrics = _metrics_for_ranking(order, rel_positions, n_relevant, ks)

    # ---- baselines -------------------------------------------------------
    n_ratings = ml.items["n_ratings"].to_numpy()[scored.item_idx]
    mean_rating = ml.items["mean_rating"].to_numpy()[scored.item_idx]
    baselines = {
        "by consensus": _metrics_for_ranking(
            np.argsort(-np.nan_to_num(mean_rating, nan=-1.0), kind="stable"),
            rel_positions, n_relevant, ks,
        ),
        "by popularity": _metrics_for_ranking(
            np.argsort(-n_ratings, kind="stable"), rel_positions, n_relevant, ks
        ),
        "random": _metrics_for_ranking(
            rng.permutation(len(scored.item_idx)), rel_positions, n_relevant, ks
        ),
    }

    # ---- predicted stars vs actual ---------------------------------------
    rating_error: dict[str, float] = {}
    if "rating" in test_df.columns:
        rows = []
        for it, actual in zip(test_df["item_idx"], test_df["rating"]):
            p = pos_of_item.get(int(it))
            if p is None or not np.isfinite(scored.rank_score[p]):
                continue
            pred_stars = np.clip(profile_mean + profile_sigma * scored.pred_z[p], 0.5, 5.0)
            rows.append((float(pred_stars), float(actual)))
        if rows:
            arr = np.array(rows)
            err = arr[:, 0] - arr[:, 1]
            rating_error = {
                "mae": float(np.abs(err).mean()),
                "rmse": float(np.sqrt((err**2).mean())),
                "n": float(len(arr)),
                "mae_trivial": float(np.abs(profile_mean - arr[:, 1]).mean()),
            }

    # ---- popularity strata ------------------------------------------------
    #
    # Computed for the baselines too, which is the comparison that decides
    # whether this project is worth anything. Overall recall@k is dominated by
    # popularity base rates -- most films anyone likes are films many people have
    # seen -- so "rank by popularity" scores well on the aggregate while being
    # useless as a recommender. The honest question is whether personalisation
    # beats it on the films that are NOT widely seen. That is this table.
    by_stratum = None
    stratum_by_method: dict[str, dict[str, float]] = {}
    if n_relevant:
        rel_list = sorted(rel_positions)
        pop = n_ratings[rel_list].astype("float64")
        qs = np.quantile(n_ratings, [0.2, 0.4, 0.6, 0.8])
        stratum = np.array([f"Q{s + 1}" for s in np.digitize(pop, qs)])

        def strata_for(ranking: np.ndarray) -> tuple[pd.DataFrame, dict[str, float]]:
            rank_of = np.empty(len(ranking), dtype="int64")
            rank_of[ranking] = np.arange(len(ranking))
            pct = 1.0 - rank_of[rel_list] / max(len(ranking) - 1, 1)
            tbl = (
                pd.DataFrame({"quintile": stratum, "percentile": pct})
                .groupby("quintile")
                .agg(n=("percentile", "size"), median_percentile=("percentile", "median"))
                .reset_index()
            )
            return tbl, dict(zip(tbl["quintile"], tbl["median_percentile"]))

        by_stratum, stratum_by_method["item-item CF"] = strata_for(order)
        for name, ranking in (
            ("by consensus", np.argsort(-np.nan_to_num(mean_rating, nan=-1.0), kind="stable")),
            ("by popularity", np.argsort(-n_ratings, kind="stable")),
        ):
            _, stratum_by_method[name] = strata_for(ranking)

    return EvalReport(
        n_train=len(train_df),
        n_test=len(test_df),
        n_relevant=n_relevant,
        n_relevant_scoreable=scoreable,
        metrics=metrics,
        baselines=baselines,
        by_stratum=by_stratum,
        rating_error=rating_error,
        stratum_by_method=stratum_by_method,
    )
