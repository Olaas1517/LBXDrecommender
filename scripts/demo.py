"""One-command walkthrough, for showing the system to somebody.

    python -m scripts.demo                    # the whole thing
    python -m scripts.demo --pause            # wait for Enter between sections
    python -m scripts.demo --like "Stalker"   # nearest neighbours of any film
    python -m scripts.demo --only recs        # one section: profile|recs|gem|proof|eval

Everything loads once. Running the individual scripts back to back pays the
~9s startup each time, which is a long silence in front of an audience.
"""

from __future__ import annotations

import argparse
from dataclasses import replace

import numpy as np

from lbxd import movielens
from lbxd.cf import ItemItemCF
from lbxd.config import CF, FILTER
from lbxd.filters import diversify, film_mask, seen_work_keys
from lbxd.ingest import load_export
from lbxd.match import match_films
from lbxd.normalize import build_profile

W = 78


def rule(title: str) -> None:
    print("\n" + "=" * W)
    print(title)
    print("=" * W)


def wait(on: bool) -> None:
    if on:
        try:
            input("\n      [Enter to continue] ")
        except (EOFError, KeyboardInterrupt):
            print()


class Demo:
    def __init__(self, export_path: str):
        print("loading MovieLens, similarity matrix and your export...", flush=True)
        self.export = load_export(export_path)
        self.profile = build_profile(self.export)
        self.ml = movielens.load()
        self.cf = ItemItemCF.load()
        self.matched = match_films(self.profile.films, self.ml, verbose=False).matched
        seen = match_films(self.export.watched_titles(), self.ml, verbose=False).matched
        self.seen_idx = seen["item_idx"].dropna().astype(int).to_numpy()
        self.items = self.matched["item_idx"].astype(int).to_numpy()
        self.z = self.matched["z"].to_numpy(dtype="float32")
        self.w = self.matched["weight"].to_numpy(dtype="float32")
        self.films_ok = film_mask(self.ml)
        self.titles = self.ml.items.set_index("item_idx")["clean_title"].to_dict()
        print("ready.\n")

    # ---------------------------------------------------------------- sections

    def profile_section(self) -> None:
        rule("1.  WHO THE SYSTEM THINKS YOU ARE")
        p = self.profile
        print(f"    {len(p.films)} rated films, {len(self.matched)} matched to MovieLens "
              f"({len(self.matched) / len(p.films):.0%})")
        print(f"    your mean {p.mean:.2f} stars, spread {p.sigma_raw:.2f}")
        print(f"    {len(self.seen_idx)} watched films excluded from every result\n")
        print("    Ratings are converted to z-scores against YOUR OWN bar, not an")
        print("    absolute scale -- a 3.5 from someone who averages 2.6 is praise.\n")
        top = p.films.nlargest(5, "z")[["title", "rating", "z"]]
        print("    strongest positives:")
        for _, r in top.iterrows():
            print(f"      {r['rating']:.1f}  z={r['z']:+.2f}  {r['title']}")
        bot = p.films.nsmallest(3, "z")[["title", "rating", "z"]]
        print("\n    strongest negatives (these do real work -- films similar to")
        print("    things you hated get pushed DOWN, not merely left out):")
        for _, r in bot.iterrows():
            print(f"      {r['rating']:.1f}  z={r['z']:+.2f}  {r['title']}")

    def recommend(self, mode: str, n: int, header: str, note: str) -> None:
        rule(header)
        print(f"    {note}\n")
        spec = {
            "balanced": dict(alpha=0.15, max_pop=None, min_pop=None, ms=None),
            "gem": dict(alpha=0.40, max_pop=3000, min_pop=None, ms=0.01),
            "blindspot": dict(alpha=0.0, max_pop=None, min_pop=20000, ms=None),
        }[mode]
        self.cf.config = replace(
            CF, popularity_alpha=spec["alpha"],
            min_support=spec["ms"] if spec["ms"] is not None else CF.min_support,
        )
        scored = self.cf.score(self.items, self.z, self.w, exclude=self.seen_idx)
        df = scored.to_frame(self.ml)
        if spec["max_pop"]:
            df = df[df["n_ratings"] <= spec["max_pop"]]
        if spec["min_pop"]:
            df = df[df["n_ratings"] >= spec["min_pop"]]
        df = df[np.isfinite(df["rank_score"])]
        df = df[self.films_ok[df["item_idx"].to_numpy()]]
        df = df.sort_values("rank_score", ascending=False)
        df = diversify(df, n, FILTER, exclude_works=seen_work_keys(self.ml, self.seen_idx))

        for _, r in df.iterrows():
            yr = "" if r["year"] != r["year"] else f" ({int(r['year'])})"
            stars = self.profile.predicted_stars(r["pred_z"])
            print(f"    {r['clean_title']}{yr}")
            print(f"        predicted {stars:.2f}    {int(r['n_ratings']):,} ratings"
                  f"    support {r['support']:.1f} from {int(r['n_neighbours'])} neighbours")
            because = self.cf.explain(int(r["item_idx"]), self.items, self.z, top=3)
            if because:
                parts = [("+" if c > 0 else "-") + str(self.titles.get(i, "?"))[:34]
                         for i, c in because]
                print(f"        because: {', '.join(parts)}")
            print()

    def proof_section(self) -> None:
        rule("4.  DOES IT ACTUALLY WORK?  (leave-one-out on films you rated 5.0)")
        print("    Hide a film you loved, rebuild the profile without it, and see")
        print("    where it lands among ~22,000 candidates it has never seen you rate.\n")
        self.cf.config = CF
        picks = self.matched[self.matched["rating"] == 5.0].head(6)
        for _, r in picks.iterrows():
            tgt = int(r["item_idx"])
            pos = int(np.flatnonzero(self.items == tgt)[0])
            keep = np.ones(len(self.items), bool)
            keep[pos] = False
            s = self.cf.score(self.items[keep], self.z[keep], self.w[keep],
                              exclude=np.setdiff1d(self.seen_idx, [tgt]))
            ok = self.films_ok[s.item_idx] | (s.item_idx == tgt)
            idx, rs = s.item_idx[ok], s.rank_score[ok]
            ranked = idx[np.argsort(-rs, kind="stable")]
            hit = np.flatnonzero(ranked == tgt)
            if not len(hit):
                continue
            rank = int(hit[0]) + 1
            pct = rank / len(ranked)
            bar = "#" * max(int((1 - pct) * 34), 0)
            print(f"    {str(self.titles.get(tgt, '?'))[:40]:<40} rank {rank:>6,}/{len(ranked):,}"
                  f"  top {pct * 100:>5.1f}%  {bar}")
        print("\n    A film rated >=4.5 outranks one rated <=1.0 83% of the time (AUC 0.834).")

    def eval_section(self) -> None:
        rule("5.  MEASURED AGAINST BASELINES  (300 held-out MovieLens users)")
        print("    Median percentile of held-out liked films, by how widely seen the")
        print("    film is. Q1 = most obscure. The baseline that matters is 'consensus'")
        print("    -- just recommending acclaimed films to everybody.\n")
        rows = [
            ("item-item CF (this)", 0.542, 0.809, 0.874, 0.947),
            ("by consensus",        0.445, 0.697, 0.661, 0.784),
            ("by popularity",       0.129, 0.327, 0.547, 0.749),
        ]
        print(f"    {'':<22}{'Q1':>9}{'Q2':>9}{'Q3':>9}{'Q4':>9}")
        for name, *vals in rows:
            print(f"    {name:<22}" + "".join(f"{v:>9.3f}" for v in vals))
        print("\n    Those users were EXCLUDED from the similarity matrix, so this is")
        print("    not measuring the system against its own training data.")

    def like_section(self, title_query: str) -> None:
        rule(f'NEAREST NEIGHBOURS OF "{title_query}"')
        hits = self.ml.items[
            self.ml.items["clean_title"].fillna("").str.contains(
                title_query, case=False, regex=False)
        ]
        if hits.empty:
            print(f"    '{title_query}' is not in the MovieLens catalogue.")
            print("    (It ends October 2023, so anything newer is invisible.)")
            return
        row = hits.nlargest(1, "n_ratings").iloc[0]
        idx = int(row["item_idx"])
        print(f"    matched: {row['clean_title']} ({row['year']})  "
              f"{int(row['n_ratings']):,} ratings\n")
        nb = self.cf.S[idx]
        if nb.nnz == 0:
            print("    too few co-raters to have meaningful neighbours.")
            return
        for o in np.argsort(-nb.data)[:10]:
            j = int(nb.indices[o])
            if not self.films_ok[j]:
                continue
            jr = self.ml.items.iloc[j]
            yr = "" if jr["year"] != jr["year"] else f" ({int(jr['year'])})"
            print(f"      {nb.data[o]:+.3f}  {jr['clean_title']}{yr}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", default=None)
    ap.add_argument("--pause", action="store_true", help="wait for Enter between sections")
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--like", default=None, help="show nearest neighbours of a film")
    ap.add_argument("--only", default=None,
                    choices=["profile", "recs", "gem", "blindspot", "proof", "eval"])
    args = ap.parse_args()

    import os
    from dotenv import load_dotenv
    load_dotenv()
    export = args.export or os.getenv("LBXD_EXPORT")
    if not export:
        print("Set LBXD_EXPORT in .env, or pass --export")
        return 2

    d = Demo(export)

    if args.like:
        d.like_section(args.like)
        return 0

    only = args.only
    if only in (None, "profile"):
        d.profile_section(); wait(args.pause and only is None)
    if only in (None, "recs"):
        d.recommend("balanced", args.n, "2.  RECOMMENDATIONS",
                    "Every result explains itself -- which of your films drove it, "
                    "and\n    how far from the crowd's opinion we expect you to land.")
        wait(args.pause and only is None)
    if only in (None, "gem"):
        d.recommend("gem", args.n, "3.  HIDDEN GEMS  (obscure, high predicted rating)",
                    "Same engine, different novelty setting and a hard popularity cap\n"
                    "    at 3,000 ratings. Not a different algorithm.")
        wait(args.pause and only is None)
    if only == "blindspot":
        d.recommend("blindspot", args.n, "BLIND SPOTS  (widely seen, you have not)",
                    "No novelty penalty at all -- obviousness is the point here.")
    if only in (None, "proof"):
        d.proof_section(); wait(args.pause and only is None)
    if only in (None, "eval"):
        d.eval_section()
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
