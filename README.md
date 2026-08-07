# Letterboxd recommender

Film recommendations from your own Letterboxd export. Personal tool first.

## Why it is built this way

Three hard constraints, verified against primary sources:

1. **No scraping.** Letterboxd's terms of use added section 6.11 in December 2025
   prohibiting automated data collection. A prominent competitor
   (`victorverma3/Letterboxd-Movie-Recommendations`) shut down because of it.
2. **No Letterboxd API.** Their access page excludes "data-analysis, visualization
   or recommendation projects", "LLM or GPT-related use", and "private or personal
   projects". This project is all three.
3. **Therefore:** the only inputs are your own export ZIP and (optionally) public
   RSS. MovieLens is the only ratings corpus available; TMDB the only metadata API.

## Setup from scratch (e.g. on a new machine)

Nothing in `data/` or `.env` is committed — both are regenerable. Full cold start
is about 6 minutes, most of it download.

```bash
git clone <this repo>
cd LBXDrecommender
py -m venv .venv
.venv\Scripts\python.exe -m pip install -e .
```

Get your Letterboxd export (Settings -> Data -> Export your data; free for all
members) and point `.env` at the unzipped folder or the ZIP itself:

```
LBXD_EXPORT=C:\path\to\letterboxd-you-2026-...
TMDB_API_KEY=
```

Download and stage MovieLens 32M (227 MB):

```bash
curl.exe -L --fail -o data/raw/ml-32m.zip https://files.grouplens.org/datasets/movielens/ml-32m.zip
tar -xf data/raw/ml-32m.zip -C data/raw
.venv\Scripts\python.exe -m lbxd.movielens      # stage (~60s, once)
.venv\Scripts\python.exe -m scripts.build_cf    # similarity matrix (~135s, once)
```

Verify it worked: `build_cf` prints the nearest neighbours of a few films at the
end. *Battle of Algiers* should come back with Wages of Fear, Throne of Blood,
Paths of Glory and other world-cinema classics. If that list looks like noise,
something is wrong with the staging and there is no point going further.

## Use

```bash
.venv\Scripts\python.exe -m scripts.profile_me              # your taste profile + match report
.venv\Scripts\python.exe -m scripts.recommend               # balanced recommendations
.venv\Scripts\python.exe -m scripts.recommend --mode gem    # obscure, high predicted rating
.venv\Scripts\python.exe -m scripts.recommend --mode blindspot  # widely seen, you have not
.venv\Scripts\python.exe -m scripts.recommend --eval        # holdout evaluation
.venv\Scripts\python.exe -m scripts.tune                    # sweep the knobs
```

## The pipeline

```
export ZIP ──▶ ingest.py ──▶ normalize.py ──▶ match.py ──▶ cf.py ──▶ ranked films
              (parse CSVs)   (z, weights)    (→ MovieLens/  (item-item
                                              TMDB ids)      similarity)
                                                  │
                                            evaluate.py (holdout, baselines, strata)
```

Read `cf.py`'s module docstring for the algorithm derivation, and `config.py` for
every tunable with its reasoning.

## Measured state (2026-07-30)

Against a real 599-rating export:

| | value |
|---|---|
| films matched to MovieLens | 439 / 599 (73.3%) |
| MovieLens corpus after filtering | 31.7M ratings, 200,948 users, 23,350 films |
| similarity build | 135 s, 4.67M nonzeros |
| predicted-star MAE | 0.881 vs 1.106 for "always guess your mean" |

Median percentile of held-out liked films, by popularity quintile
(10 pooled random splits; Q1 = least widely seen):

| | Q1 | Q2 | Q3 | Q4 | Q5 |
|---|---|---|---|---|---|
| n | 9 | 13 | 18 | 29 | 223 |
| **item-item CF** | 0.282 | **0.773** | **0.760** | **0.744** | 0.950 |
| by consensus | 0.277 | 0.763 | 0.756 | 0.693 | 0.935 |
| by popularity | 0.049 | 0.253 | 0.411 | 0.599 | **0.980** |

### How to read that

- **CF triples the popularity baseline on the mid-tail (Q2–Q4).** That is the
  win, and it is where a recommender earns its keep.
- **Aggregate recall@k is misleading here.** "By popularity" wins it (R@100 0.225
  vs 0.050) purely because 223 of 292 relevant held-out films sit in Q5. Ranking
  by rating count will always score well against a popularity-skewed test set
  while being useless as a recommender. Always read the stratified table.
- **CF does not yet meaningfully beat "by consensus".** 0.773 vs 0.763 is noise.
  This is the project's central open problem: personalisation that ties with
  "recommend acclaimed films to everyone" is decorative.

## Known problems

1. **Predictions are overconfident.** Everything comes back 4.6–4.96 stars,
   because a weighted average over 2–5 neighbours just reproduces those
   neighbours' z-scores. Needs support-based shrinkage on `pred_z`.
2. **Ties with the consensus baseline.** `match.py` computes `residual`
   (your rating minus the crowd's) and scoring never uses it. To beat consensus
   the signal has to encode *deviation* from consensus.
3. **TV leaks in.** "His Last Vow ()" is a Sherlock episode. IMDb's
   `title.basics.titleType` is the clean fix.
4. **No same-work dedup.** Rating one part of Bondarchuk's *War and Peace*
   returns the other three parts as top recommendations.
5. **Q1 is unreachable** (0.28 for every method). Films with 20–60 MovieLens
   ratings cannot accumulate enough co-rating support to rank. Genuine obscurity
   needs the content-based track, not CF.
6. **26.7% of the library does not match.** Breakdown of the 160 misses: ~60
   post-2023 (MovieLens ends Oct 2023), ~50 TV episodes logged as films
   (Love Death & Robots, Sherlock), ~30 Soviet/Kazakh films absent from the
   catalogue, the rest genuinely missing. MovieLens titles are also unreliable —
   *Endgame* is filed as "Avengers: Infinity War - Part II".

## Where to pick up

In priority order. (1) and (2) are the ones that decide whether this project has
a point — everything else is polish on top of a ranking that currently ties with
"recommend acclaimed films to everybody".

1. **Score on the consensus residual, not only z.** `match.py` already computes
   `residual = your_rating - crowd_average` and the scorer ignores it. This is
   the most likely reason CF only ties the consensus baseline: to beat consensus
   the signal has to encode *disagreement* with it. Try residual as the CF input
   value, and a z/residual blend. Measure on the harness, do not eyeball.
2. **Shrink `pred_z` when support is low.** Predictions currently saturate at
   4.6–4.96 stars because averaging 2–5 neighbours just reproduces their z.
   Apply the same trick used on similarity: `pred_z *= support / (support + k)`.
3. **Filter TV, dedup same-work franchises.** IMDb `title.basics.titleType`.
4. **IMDb `title.akas` for matching**, then TMDB search as the real identity
   resolver. Free, no key needed for akas.
5. **TMDB content track** for post-2023 films, which MovieLens cannot see.
6. **LLM taste profile** from the review text, and the written pitch.

Re-run `scripts.tune` after any scoring change. The number that matters is the
stratified table, not aggregate recall.

## Data sources considered

| Source | Per-user ratings? | Use |
|---|---|---|
| MovieLens 32M | **yes** — the only option | CF substrate, tmdbId mapping |
| MovieLens ml-latest | yes; 330k users but catalogue ends Jul 2023 and it is a shifting dev dataset | possible density upgrade |
| IMDb non-commercial | **no** — aggregate ratings only | `title.akas` for matching, crew for content model, `titleType` for TV filtering |
| TMDB API | no | metadata, keywords, watch providers, behaviour-derived `similar`/`recommendations` for post-2023 films |
| Netflix Prize | yes, but 17,770 films all pre-2005, withdrawn after the de-anonymisation suit | no |
