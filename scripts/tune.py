"""Sweep the scoring knobs on YOUR export and report what each one costs.

    python -m scripts.tune

READ THIS BEFORE TUNING ANYTHING ON THE OUTPUT.

One export puts roughly 11-17 held-out liked films in each of the lower
popularity quintiles. Medians over eleven observations move around by 0.2 for no
reason at all, and every knob in this project was once chosen by comparing
numbers of exactly that kind. Use `scripts.eval_synthetic` to CHOOSE parameters
-- 300 users and 20,000 observations -- and use this script to see how a choice
lands on your particular library.

The `rankable` column is the one to read first. A configuration that posts a
beautiful median while ranking a third of the candidates is not better than one
that posts a worse median while ranking all of them; it is answering an easier
question. Films the engine refuses to rank are tied at -inf, and their
"percentile" is a tie-break artefact rather than a result.
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

    # The knobs that change the answer, and the two baselines they must beat.
    grid = [
        ("BEFORE (original settings)", replace(
            CF, use_residual_input=False, consensus_beta=0.0, pred_shrinkage_k=0.0,
            rank_shrinkage_k=0.0, min_support=0.1, pred_gain=1.0)),
        ("current defaults", CF),
        ("min_support 0.10 (old floor)", replace(CF, min_support=0.10)),
        ("min_support 0.01 (gem mode)", replace(CF, min_support=0.01)),
        ("no rank shrinkage", replace(CF, rank_shrinkage_k=0.0)),
        ("beta 0.5 (half consensus)", replace(CF, consensus_beta=0.5)),
        ("beta 0.0 (pure disagreement)", replace(CF, consensus_beta=0.0)),
        ("alpha 0.0 (no novelty penalty)", replace(CF, popularity_alpha=0.0)),
        ("alpha 0.4 (aggressive novelty)", replace(CF, popularity_alpha=0.4)),
    ]

    rows = []
    baseline_row = None
    for name, cfg in grid:
        cf.config = cfg
        rep = evaluate(
            matched, cf, ml, profile.mean, profile.sigma,
            exclude_item_idx=seen_idx, n_repeats=10,
        )
        strata = rep.stratum_by_method.get("item-item CF", {})
        rows.append(
            {
                "config": name,
                "rankable": rep.n_relevant_scoreable / max(rep.n_relevant, 1),
                "med_pct": rep.metrics["median_percentile"],
                "R@50": rep.metrics["recall@50"],
                "MAE": rep.rating_error.get("mae", float("nan")),
                "Q1-3": np.nanmean([strata.get(f"Q{i}", np.nan) for i in (1, 2, 3)]),
                "Q4": strata.get("Q4", float("nan")),
                "Q5": strata.get("Q5", float("nan")),
            }
        )
        if baseline_row is None:
            b = rep.baselines["by popularity"]
            c = rep.baselines["by consensus"]
            bs = rep.stratum_by_method["by consensus"]
            baseline_row = (
                f"  BASELINE by popularity : med_pct {b['median_percentile']:.3f}  "
                f"R@50 {b['recall@50']:.3f}\n"
                f"  BASELINE by consensus  : med_pct {c['median_percentile']:.3f}  "
                f"R@50 {c['recall@50']:.3f}   "
                f"Q1-3 {np.nanmean([bs.get(f'Q{i}', np.nan) for i in (1, 2, 3)]):.3f}  "
                f"Q4 {bs.get('Q4', float('nan')):.3f}"
            )
            n_by_q = dict(zip(rep.by_stratum["quintile"], rep.by_stratum["n"].astype(int)))
            print(f"  held-out liked films per quintile: {n_by_q}")
        print(f"  swept: {name}", flush=True)

    print()
    print("=" * 92)
    print("SWEEP RESULTS  (10 pooled random splits each)")
    print("=" * 92)
    print(baseline_row)
    print()
    df = pd.DataFrame(rows)
    print(df.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print()
    print("rankable = fraction of held-out liked films the engine will score at all.")
    print("           READ THIS FIRST. A high med_pct with low coverage is not a")
    print("           better system, it is an easier question.")
    print("Q1-3     = median percentile on the LESS popular films (the valuable cases)")
    print("Q5       = median percentile on the most popular films (the easy ones)")
    print()
    print("Cells below Q4 rest on ~11-17 films. To CHOOSE a setting rather than")
    print("just look at one, use: python -m scripts.eval_synthetic --compare")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
