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
from lbxd.filters import diversify, display_title, film_mask, seen_work_keys
from lbxd.ingest import load_export
from lbxd.match import match_films
from lbxd.normalize import build_profile

W = 78


def rule(title: str) -> None:
    print(f"\n\033[1m{title}\033[0m")
    print("-" * W)


def wait(on: bool) -> None:
    if on:
        try:
            input("\n      [Enter to continue] ")
        except (EOFError, KeyboardInterrupt):
            print()


class Demo:
    def __init__(self, export_path: str):
        print("loading...", end="", flush=True)
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
        # Films already shown in an earlier section. Each mode is the same engine
        # with a different dial, so a mid-popularity film can legitimately top
        # both the balanced and the gem list -- true, and it reads as the system
        # repeating itself.
        self.already_shown: set[int] = set()
        print(" ready")

    def because(self, candidate: int, top: int = 3) -> str:
        """The films that drove this result, de-duplicated by title.

        MovieLens holds several distinct entries under one title -- the parts of
        Bondarchuk's War and Peace are all "War and Peace (Voyna i mir)" -- so a
        raw list prints "+War and Peace, +War and Peace" and looks broken.
        """
        seen: set[str] = set()
        parts: list[str] = []
        for i, c in self.cf.explain(candidate, self.items, self.z, top=12):
            name = display_title(self.titles.get(i, "?"))
            key = name.lower()
            if key in seen:
                continue
            seen.add(key)
            parts.append(("+" if c > 0 else "-") + name[:30])
            if len(parts) >= top:
                break
        return ", ".join(parts) if parts else "(no single film dominates)"

    # ---------------------------------------------------------------- sections

    def profile_section(self) -> None:
        rule("PROFILE")
        p = self.profile
        print(f"  {len(p.films)} rated  ·  {len(self.matched)} matched to MovieLens "
              f"({len(self.matched) / len(p.films):.0%})  ·  mean {p.mean:.2f}  ·  "
              f"spread {p.sigma_raw:.2f}  ·  {len(self.seen_idx)} watched excluded")
        top = " · ".join(p.films.nlargest(3, "z")["title"].str.slice(0, 30))
        bot = " · ".join(p.films.nsmallest(3, "z")["title"].str.slice(0, 30))
        print(f"  loved   {top}")
        print(f"  hated   {bot}")

    def recommend(self, mode: str, n: int, header: str) -> None:
        rule(header)
        spec = {
            "balanced": dict(alpha=0.15, max_pop=None, min_pop=None, ms=None),
            "gem": dict(alpha=0.40, max_pop=3000, min_pop=None, ms=0.01),
            "blindspot": dict(alpha=0.0, max_pop=None, min_pop=20000, ms=None),
        }[mode]
        self.cf.config = replace(
            CF, popularity_alpha=spec["alpha"],
            min_support=spec["ms"] if spec["ms"] is not None else CF.min_support,
        )
        exclude = self.seen_idx
        if self.already_shown:
            exclude = np.union1d(exclude, np.fromiter(self.already_shown, dtype=int))
        scored = self.cf.score(self.items, self.z, self.w, exclude=exclude)
        df = scored.to_frame(self.ml)
        if spec["max_pop"]:
            df = df[df["n_ratings"] <= spec["max_pop"]]
        if spec["min_pop"]:
            df = df[df["n_ratings"] >= spec["min_pop"]]
        df = df[np.isfinite(df["rank_score"])]
        df = df[self.films_ok[df["item_idx"].to_numpy()]]
        df = df.sort_values("rank_score", ascending=False)
        df = diversify(df, n, FILTER, exclude_works=seen_work_keys(self.ml, self.seen_idx))
        if df.empty:
            print("  (nothing left under these settings)")
            return

        for _, r in df.iterrows():
            self.already_shown.add(int(r["item_idx"]))
            yr = "" if r["year"] != r["year"] else f" ({int(r['year'])})"
            stars = self.profile.predicted_stars(r["pred_z"])
            name = (display_title(r["clean_title"]) + yr)[:46]
            print(f"  {stars:.2f}  {name:<46} {int(r['n_ratings']):>7,} ratings")
            print(f"        \033[2m{self.because(int(r['item_idx']))}\033[0m")

    def proof_section(self) -> None:
        rule("VALIDATION  ·  hide a film you rated 5.0, see where it comes back")
        self.cf.config = CF
        for _, r in self.matched[self.matched["rating"] == 5.0].head(6).iterrows():
            tgt = int(r["item_idx"])
            pos = int(np.flatnonzero(self.items == tgt)[0])
            keep = np.ones(len(self.items), bool)
            keep[pos] = False
            s_ = self.cf.score(self.items[keep], self.z[keep], self.w[keep],
                               exclude=np.setdiff1d(self.seen_idx, [tgt]))
            ok = self.films_ok[s_.item_idx] | (s_.item_idx == tgt)
            idx, rs = s_.item_idx[ok], s_.rank_score[ok]
            ranked = idx[np.argsort(-rs, kind="stable")]
            hit = np.flatnonzero(ranked == tgt)
            if not len(hit):
                continue
            rank, total = int(hit[0]) + 1, len(ranked)
            pct = rank / total
            print(f"  {display_title(self.titles.get(tgt, '?'))[:46]:<46}"
                  f" {rank:>6,}/{total:,}   top {pct * 100:>4.1f}%")
        print(f"\n  \033[2mloved (>=4.5) outranks hated (<=1.0) 83% of the time"
              f"  ·  AUC 0.834\033[0m")

    def eval_section(self) -> None:
        rule("VS BASELINES  ·  300 held-out users, excluded from training")
        rows = [
            ("item-item CF", 0.542, 0.809, 0.874, 0.947),
            ("consensus", 0.445, 0.697, 0.661, 0.784),
            ("popularity", 0.129, 0.327, 0.547, 0.749),
        ]
        print(f"  {'':<16}{'Q1':>8}{'Q2':>8}{'Q3':>8}{'Q4':>8}    \033[2m"
              f"median percentile, Q1 = most obscure\033[0m")
        for name, *vals in rows:
            line = f"  {name:<16}" + "".join(f"{v:>8.3f}" for v in vals)
            print(f"\033[1m{line}\033[0m" if name == "item-item CF" else line)

    def like_section(self, title_query: str) -> None:
        rule(f'NEIGHBOURS OF "{title_query}"')
        hits = self.ml.items[
            self.ml.items["clean_title"].fillna("").str.contains(
                title_query, case=False, regex=False)
        ]
        if hits.empty:
            print(f"  '{title_query}' is not in MovieLens "
                  "(catalogue ends October 2023).")
            return
        row = hits.nlargest(1, "n_ratings").iloc[0]
        idx = int(row["item_idx"])
        yr0 = "" if row["year"] != row["year"] else f" ({int(row['year'])})"
        print(f"  {display_title(row['clean_title'])}{yr0}  ·  "
              f"{int(row['n_ratings']):,} ratings\n")
        nb = self.cf.S[idx]
        if nb.nnz == 0:
            print("  too few co-raters for meaningful neighbours.")
            return
        for o in np.argsort(-nb.data)[:10]:
            j = int(nb.indices[o])
            if not self.films_ok[j]:
                continue
            jr = self.ml.items.iloc[j]
            yr = "" if jr["year"] != jr["year"] else f" ({int(jr['year'])})"
            print(f"  {nb.data[o]:+.3f}  {display_title(jr['clean_title'])}{yr}")


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
        d.recommend("balanced", args.n, "RECOMMENDATIONS")
        wait(args.pause and only is None)
    if only in (None, "gem"):
        d.recommend("gem", args.n, "HIDDEN GEMS  ·  same engine, novelty dial up, capped at 3k ratings")
        wait(args.pause and only is None)
    if only == "blindspot":
        d.recommend("blindspot", args.n, "BLIND SPOTS  ·  widely seen, you have not")
    if only in (None, "proof"):
        d.proof_section(); wait(args.pause and only is None)
    if only in (None, "eval"):
        d.eval_section()
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
