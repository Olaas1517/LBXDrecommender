"""Sweep the scoring knobs and report what each one costs and buys.

    py -m scripts.tune

The point is diagnostic separation. The first evaluation run showed the engine
losing to a dumb "recommend whatever is most-rated" baseline, and there are two
completely different possible causes:

  (a) the collaborative signal is weak, or
  (b) the novelty penalty is deliberately pushing down popular films, and the
      held-out ground truth happens to be popularity-skewed, so we are being
      punished by the metric for doing what we asked it to do.

Setting popularity_alpha to 0 distinguishes these. If the engine beats the
popularity baseline at alpha=0, the signal is fine and the penalty is simply
expensive -- a product decision, not a bug. If it still loses, the retrieval is
broken and no amount of tuning will save it.
"""

from __future__ import annotations

import os
from dataclasses import replace

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from lbxd import movielens
from lbxd.cf import ItemItemCF
from lbxd.config import CF, EVAL, NormalizeConfig
from lbxd.evaluate import evaluate
from lbxd.ingest import load_export
from lbxd.match import match_films
from lbxd.normalize import build_profile


def main() -> int:
    load_dotenv()
    export = load_export(os.environ["LBXD_EXPORT"])
    profile = build_profile(export, NormalizeConfig())
    ml = movielens.load()
    matched = match_films(profile.films, ml, verbose=False).matched
    seen = match_films(export.watched_titles(), ml, verbose=False).matched
    seen_idx = seen["item_idx"].dropna().astype(int).to_numpy()

    cf = ItemItemCF.load()

    grid = [
        (alpha, ms)
        for alpha in (0.0, 0.15, 0.40)
        for ms in (1.0, 0.3, 0.1)
    ]

    rows = []
    baseline_row = None
    for alpha, ms in grid:
        cf.config = replace(CF, popularity_alpha=alpha, min_support=ms)
        rep = evaluate(
            matched, cf, ml, profile.mean, profile.sigma,
            exclude_item_idx=seen_idx, n_repeats=10,
        )
        strata = {}
        if rep.by_stratum is not None:
            strata = dict(zip(rep.by_stratum["quintile"], rep.by_stratum["median_percentile"]))
        rows.append(
            {
                "alpha": alpha,
                "min_sup": ms,
                "scoreable": rep.n_relevant_scoreable / max(rep.n_relevant, 1),
                "med_pct": rep.metrics["median_percentile"],
                "R@50": rep.metrics["recall@50"],
                "R@100": rep.metrics["recall@100"],
                "MAE": rep.rating_error.get("mae", float("nan")),
                "Q1-3": np.nanmean([strata.get(f"Q{i}", np.nan) for i in (1, 2, 3)]),
                "Q5": strata.get("Q5", float("nan")),
            }
        )
        if baseline_row is None:
            b = rep.baselines["by popularity"]
            c = rep.baselines["by consensus"]
            baseline_row = (
                f"  BASELINE by popularity : med_pct {b['median_percentile']:.3f}  "
                f"R@50 {b['recall@50']:.3f}  R@100 {b['recall@100']:.3f}\n"
                f"  BASELINE by consensus  : med_pct {c['median_percentile']:.3f}  "
                f"R@50 {c['recall@50']:.3f}  R@100 {c['recall@100']:.3f}"
            )
        print(f"  swept alpha={alpha} min_support={ms}")

    print()
    print("=" * 92)
    print("SWEEP RESULTS  (10 pooled random splits each)")
    print("=" * 92)
    print(baseline_row)
    print()
    df = pd.DataFrame(rows)
    print(df.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print()
    print("alpha    = novelty penalty, z-units per 10x popularity")
    print("min_sup  = similarity mass required before we will rank a film at all")
    print("scoreable= fraction of held-out liked films the engine can score")
    print("Q1-3     = median percentile on the LESS popular films (the hard, valuable cases)")
    print("Q5       = median percentile on the most popular films (the easy cases)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
