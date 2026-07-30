"""Stage 0 + matching, end to end, on a real Letterboxd export.

    py -m scripts.profile_me --export "C:\\path\\to\\letterboxd-you-..."

Writes data/processed/profile_matched.csv, which is what the CF engine consumes.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from lbxd import movielens
from lbxd.config import DATA_PROCESSED, NormalizeConfig
from lbxd.ingest import load_export
from lbxd.match import match_films
from lbxd.normalize import build_profile


def main() -> int:
    load_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", default=os.getenv("LBXD_EXPORT"), help="export .zip or folder")
    ap.add_argument("--use-likes", action="store_true", help="let film likes nudge the signal")
    ap.add_argument("--no-recency", action="store_true", help="disable recency decay")
    args = ap.parse_args()

    if not args.export:
        print("Need --export (or set LBXD_EXPORT in .env)", file=sys.stderr)
        return 2

    print("=" * 70)
    print("1. READING EXPORT")
    print("=" * 70)
    export = load_export(args.export)
    print(export.summary())

    print()
    print("=" * 70)
    print("2. STAGE 0 - NORMALIZATION")
    print("=" * 70)
    cfg = NormalizeConfig(
        use_likes=args.use_likes,
        recency_half_life_years=None if args.no_recency else 3.0,
    )
    profile = build_profile(export, cfg)
    print(profile.summary())

    print()
    print("=" * 70)
    print("3. MATCHING TO MOVIELENS / TMDB")
    print("=" * 70)
    ml = movielens.load()
    result = match_films(profile.films, ml)

    if not result.unmatched.empty:
        DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
        miss_path = DATA_PROCESSED / "unmatched.csv"
        result.unmatched.to_csv(miss_path, index=False, encoding="utf-8")
        print(f"\nunmatched written to {miss_path}")
        print("a sample of what we could not find:")
        for _, row in result.unmatched.head(15).iterrows():
            yr = "" if row.get("year") is None else row.get("year")
            print(f"  {row['title']} ({yr})")

    out = DATA_PROCESSED / "profile_matched.csv"
    result.matched.to_csv(out, index=False, encoding="utf-8")
    print(f"\nmatched profile written to {out}")

    # Coverage check that actually matters: how much of this person's history is
    # invisible to MovieLens because it postdates the dataset?
    films = profile.films
    if "year" in films.columns:
        post = films[films["year"] >= 2024]
        print(
            f"\nfilms you rated from 2024 onward: {len(post)} "
            f"({len(post) / max(len(films), 1):.1%}) "
            "-- MovieLens cannot see any of these (data ends Oct 2023)"
        )

    # Also report how much of the *watchlist* we can resolve, since the watchlist
    # is a retrieval source later on.
    wl = export.watchlist
    if not wl.empty:
        print()
        print("watchlist matching:")
        match_films(wl, ml)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
