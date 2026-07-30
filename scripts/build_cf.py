"""Precompute the item-item similarity matrix. Run once (minutes), reuse forever.

    py -m scripts.build_cf
"""

from __future__ import annotations

import time

from lbxd import movielens
from lbxd.cf import build_similarity, save_similarity
from lbxd.config import CF


def main() -> int:
    print("loading staged MovieLens...")
    ml = movielens.load()
    print(ml.summary())

    print()
    print(
        f"building item-item similarity  "
        f"(lambda={CF.shrinkage_lambda}, top_k={CF.top_k_neighbours})"
    )
    t0 = time.perf_counter()
    S = build_similarity(ml, CF)
    print(f"done in {time.perf_counter() - t0:.1f}s")

    save_similarity(S, ml)
    print("saved.")

    # Smoke test: a film's nearest neighbours should be obviously related. If
    # this list looks like noise, the shrinkage or the centring is wrong, and
    # there is no point continuing to the eval harness.
    import numpy as np

    items = ml.items
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
