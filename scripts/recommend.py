"""Generate recommendations, and evaluate them.

    py -m scripts.recommend                 # balanced
    py -m scripts.recommend --mode gem      # obscure, high predicted rating
    py -m scripts.recommend --mode blindspot  # widely seen, you somehow have not
    py -m scripts.recommend --eval          # run the holdout evaluation instead
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import replace

import numpy as np
from dotenv import load_dotenv

from lbxd import movielens
from lbxd.cf import ItemItemCF
from lbxd.config import CF, NormalizeConfig
from lbxd.evaluate import evaluate
from lbxd.ingest import load_export
from lbxd.match import match_films
from lbxd.normalize import build_profile

# The retrieval modes from the design. Each is the same engine with a different
# novelty setting and popularity window -- not different algorithms.
MODES = {
    # Default: mild novelty preference, no popularity window.
    "balanced": dict(alpha=0.15, max_pop=None, min_pop=None),
    # Hidden gems: pay real predicted-rating cost for obscurity, and hard-cap
    # popularity so crowd-pleasers cannot appear at all.
    "gem": dict(alpha=0.40, max_pop=3000, min_pop=None),
    # Blind spots: films a lot of people have seen and you have not. No novelty
    # penalty at all -- obviousness is the point of this mode.
    "blindspot": dict(alpha=0.0, max_pop=None, min_pop=20000),
}


def main() -> int:
    load_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", default=os.getenv("LBXD_EXPORT"))
    ap.add_argument("--mode", choices=sorted(MODES), default="balanced")
    ap.add_argument("--n", type=int, default=15)
    ap.add_argument("--eval", action="store_true", help="run holdout evaluation")
    ap.add_argument("--no-recency", action="store_true")
    ap.add_argument("--alpha", type=float, default=None, help="override novelty penalty")
    ap.add_argument("--min-support", type=float, default=None, help="override support floor")
    args = ap.parse_args()

    if not args.export:
        print("Need --export (or LBXD_EXPORT in .env)", file=sys.stderr)
        return 2

    export = load_export(args.export)
    profile = build_profile(
        export,
        NormalizeConfig(recency_half_life_years=None if args.no_recency else 3.0),
    )
    ml = movielens.load()

    print(f"matching your {len(profile.films)} rated films...")
    matched = match_films(profile.films, ml, verbose=False).matched
    print(f"  {len(matched)} matched to MovieLens")

    # The already-seen filter must cover everything watched, not just everything
    # rated. Recommending a film someone watched years ago and never rated is a
    # visible, trust-destroying error.
    print("matching your full watch history (for the exclusion list)...")
    seen = match_films(export.watched_titles(), ml, verbose=False).matched
    seen_idx = seen["item_idx"].dropna().astype(int).to_numpy() if not seen.empty else np.array([], int)
    print(f"  {len(seen_idx)} watched films identified and will be excluded")

    rated_items = matched["item_idx"].astype(int).to_numpy()
    z = matched["z"].to_numpy(dtype="float32")
    w = matched["weight"].to_numpy(dtype="float32")

    if args.eval:
        cf = ItemItemCF.load()
        cf.config = replace(
            CF,
            popularity_alpha=CF.popularity_alpha if args.alpha is None else args.alpha,
            min_support=CF.min_support if args.min_support is None else args.min_support,
        )
        print()
        print("=" * 70)
        print("HOLDOUT EVALUATION")
        print("=" * 70)
        report = evaluate(
            matched, cf, ml, profile.mean, profile.sigma, exclude_item_idx=seen_idx
        )
        print(report.summary())
        return 0

    spec = MODES[args.mode]
    cf = ItemItemCF.load()
    # The novelty dial is a scoring-time parameter, so switching modes needs no
    # recompute -- just a different config on the same loaded matrix.
    cf.config = replace(CF, popularity_alpha=spec["alpha"])

    scored = cf.score(rated_items, z, w, exclude=seen_idx)
    df = scored.to_frame(ml)

    if spec["max_pop"]:
        df = df[df["n_ratings"] <= spec["max_pop"]]
    if spec["min_pop"]:
        df = df[df["n_ratings"] >= spec["min_pop"]]
    df = df[np.isfinite(df["rank_score"])]
    df = df.sort_values("rank_score", ascending=False).head(args.n)

    print()
    print("=" * 78)
    print(f"MODE: {args.mode}   (novelty penalty {spec['alpha']}, "
          f"popularity window {spec['min_pop'] or 0}-{spec['max_pop'] or 'inf'})")
    print("=" * 78)
    for _, r in df.iterrows():
        stars = profile.to_stars(r["pred_z"])
        yr = "" if r["year"] != r["year"] else int(r["year"])
        print(f"\n  {r['clean_title']} ({yr})")
        print(
            f"    predicted {stars:.2f} stars (your mean is {profile.mean:.2f})"
            f"  |  {int(r['n_ratings']):,} MovieLens ratings"
            f"  |  support {r['support']:.1f} from {int(r['n_neighbours'])} neighbours"
        )
        because = cf.explain(int(r["item_idx"]), rated_items, z, top=3)
        if because:
            parts = []
            for item_idx, contrib in because:
                row = ml.items.iloc[item_idx]
                direction = "+" if contrib > 0 else "-"
                parts.append(f"{direction}{row['clean_title']}")
            print(f"    because: {', '.join(parts)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
