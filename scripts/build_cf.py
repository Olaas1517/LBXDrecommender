"""Precompute the item-item similarity matrix. Run once (minutes), reuse forever.

    python -m scripts.build_cf

Also writes two things the scorer needs alongside the matrix:

  item_stats.npz    per-film popularity, mean rating, and mean z-score. The last
                    of these is the consensus baseline that CFConfig's
                    residual decomposition subtracts out and adds back.
  holdout_users.npy the MovieLens users deliberately EXCLUDED from the build, so
                    that scripts.eval_synthetic can score them without testing
                    the system against its own training data.

Excluding a few hundred raters out of 200,948 costs the production matrix
nothing measurable, and it is what makes the headline numbers honest, so it is
done unconditionally rather than behind a flag.
"""

from __future__ import annotations

import time

import numpy as np

from lbxd import movielens
from lbxd.cf import build_similarity, item_z_means, save_similarity, user_zscores
from lbxd.config import CF, DATA_PROCESSED
from lbxd.synthetic import HOLDOUT_USERS_NPY, pick_holdout_users


def main() -> int:
    print("loading staged MovieLens...")
    ml = movielens.load()
    print(ml.summary())

    holdout = pick_holdout_users(ml)
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    np.save(DATA_PROCESSED / HOLDOUT_USERS_NPY, holdout)
    print(f"\nholding out {len(holdout)} users for evaluation (excluded from the build)")

    print()
    print(
        f"building item-item similarity  "
        f"(lambda={CF.shrinkage_lambda}, top_k={CF.top_k_neighbours})"
    )
    t0 = time.perf_counter()
    S = build_similarity(ml, CF, exclude_users=holdout)
    print(f"done in {time.perf_counter() - t0:.1f}s")

    # The consensus baseline, computed from the same z-scores the matrix used.
    z = user_zscores(ml)
    z_mean = item_z_means(ml, z, CF)
    save_similarity(S, ml, z_mean)
    print("saved.")

    items = ml.items
    print(
        f"\nconsensus z ranges {z_mean.min():+.3f} to {z_mean.max():+.3f} "
        f"(sd {z_mean.std():.3f})"
    )
    best = np.argsort(-z_mean)[:5]
    print("highest-consensus films (these are what 'by consensus' recommends):")
    for j in best:
        r = items.iloc[int(j)]
        print(f"   {z_mean[j]:+.3f}  {r['clean_title']} ({r['year']})  n={r['n_ratings']:,}")

    # Smoke test: a film's nearest neighbours should be obviously related. If
    # this list looks like noise, the shrinkage or the centring is wrong, and
    # there is no point continuing to the eval harness.
    for title_probe in ["Apocalypse Now", "Battle of Algiers", "Lego Movie"]:
        hits = items[items["clean_title"].fillna("").str.contains(title_probe, case=False, regex=False)]
        if hits.empty:
            continue
        row = hits.iloc[0]
        idx = int(row["item_idx"])
        nb = S[idx]
        order = np.argsort(-nb.data)[:8]
        print(f"\nnearest neighbours of {row['clean_title']!r}:")
        for o in order:
            j = int(nb.indices[o])
            jr = items.iloc[j]
            print(f"   {nb.data[o]:+.3f}  {jr['clean_title']} ({jr['year']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
