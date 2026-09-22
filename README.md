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

## Setup from scratch

Nothing in `data/` or `.env` is committed — both are regenerable, and both hold
personal data. Full cold start is about 6 minutes, most of it download.

```bash
git clone <this repo>
cd LBXDrecommender
python3.11 -m venv .venv
.venv/bin/python -m pip install -e .
```

On Windows use `py -m venv .venv` and `.venv\Scripts\python.exe` in place of
`.venv/bin/python` throughout. Python 3.11 or newer is required.

Get your Letterboxd export (Settings → Data → Export your data; free for all
members) and point `.env` at the ZIP or the unzipped folder:

```
LBXD_EXPORT=data/raw/letterboxd-you-2026-...zip
TMDB_API_KEY=
```

Download and stage MovieLens 32M (227 MB):

```bash
curl -L --fail -o data/raw/ml-32m.zip https://files.grouplens.org/datasets/movielens/ml-32m.zip
tar -xf data/raw/ml-32m.zip -C data/raw
.venv/bin/python -m lbxd.movielens      # stage (~60s, once)
.venv/bin/python -m scripts.build_cf    # similarity matrix (~80s, once)
```

`build_cf` prints the nearest neighbours of a few films at the end. *Battle of
Algiers* should come back with Wages of Fear, Throne of Blood, Paths of Glory and
other world-cinema classics. If that list looks like noise, the staging is wrong
and there is no point going further.

The first recommendation run also downloads IMDb's `title.basics` (~200 MB) to
tell films from television. It is cached; you pay for it once.

## Use

```bash
.venv/bin/python -m scripts.profile_me              # your taste profile + match report
.venv/bin/python -m scripts.recommend               # balanced recommendations
.venv/bin/python -m scripts.recommend --mode gem    # obscure, high predicted rating
.venv/bin/python -m scripts.recommend --mode blindspot  # widely seen, you have not
.venv/bin/python -m scripts.recommend --no-shorts   # feature films only
.venv/bin/python -m scripts.recommend --only-shorts # short films only
.venv/bin/python -m scripts.recommend --eval        # holdout evaluation on YOUR ratings
.venv/bin/python -m scripts.eval_synthetic --compare # evaluation on 300 other people
.venv/bin/python -m scripts.tune                    # sweep the knobs
.venv/bin/python -m pytest                          # 83 tests, no data required
```

## The pipeline

```
export ZIP ──▶ ingest.py ──▶ normalize.py ──▶ match.py ──▶ cf.py ──▶ filters.py ──▶ ranked films
              (parse CSVs)   (z, weights)    (→ MovieLens/  (item-item  (TV, shorts,
                                              TMDB ids)      similarity)  same-work)
                                                  │
                                            evaluate.py (holdout, baselines, strata)
                                            synthetic.py (the same harness, on 300
                                                          held-out MovieLens users)
```

Read `cf.py`'s module docstring for the algorithm derivation, and `config.py` for
every tunable with the measurement behind it.

## Measured state (2026-09-22)

Against a real 605-rating export: **431 matched to MovieLens (71.2%)**, 451
watched films excluded from results.

### The headline

The project's central open problem was: *"CF does not meaningfully beat 'by
consensus'. Personalisation that ties with 'recommend acclaimed films to
everyone' is decorative."*

**It is no longer true.** Measured on 300 held-out MovieLens users (20,545
held-out liked films, pooled over 3 splits each; these users were excluded from
the similarity build, so this is not testing against training data):

Median percentile of held-out liked films, by popularity quintile.
Q1 = least widely seen. Higher is better; 0.50 is a coin flip.

| | Q1 | Q2 | Q3 | Q4 | Q5 |
|---|---|---|---|---|---|
| n | 38 | 80 | 222 | 961 | 19,244 |
| **item-item CF (now)** | **0.542** | **0.809** | **0.874** | **0.947** | 0.962 |
| item-item CF (before) | 0.469 | 0.467 | 0.670 | 0.957 | 0.972 |
| by consensus | 0.445 | 0.697 | 0.661 | 0.784 | 0.924 |
| by popularity | 0.129 | 0.327 | 0.547 | 0.749 | **0.988** |

The old configuration *lost* to consensus on Q2 (0.467 vs 0.697) and tied on Q3.
It now wins on every stratum except Q5, where popularity is unbeatable by
construction and nobody needs help anyway.

### What actually fixed it

Not what the previous notes predicted. The three findings, in order of how wrong
the prior diagnosis was:

**1. The problem was coverage, not signal.** Splitting "how well does it rank?"
from "will it rank this at all?" was the single most useful change to the
harness. Where the engine was willing to make a claim it was already excellent —
0.97-0.99 across every quintile, before any of this work. It simply refused to
answer about most non-popular films:

| fraction the engine will rank at all | Q1 | Q2 | Q3 | Q4 | Q5 |
|---|---|---|---|---|---|
| before | 0.158 | 0.200 | 0.329 | 0.675 | 0.980 |
| after | **0.395** | **0.525** | **0.568** | **0.838** | 0.992 |

Films below `min_support` are all tied at `-inf`, so their "percentile" was
whatever the tie-break happened to give them. The old stratified table was
averaging real rankings together with tie-break positions and reporting the
result as a measurement. `min_support` went 0.1 → 0.03 (and 0.01 in `gem` mode);
the table above is the whole justification.

**2. Predictions were under-confident, not overconfident.** The note said
"everything comes back 4.6–4.96 stars, needs support-based shrinkage". Measured
on 8,647 held-out ratings:

```
actual_z = 1.172 * pred_z - 0.046      sd(pred) 0.488  vs  sd(actual) 0.985
```

A slope above 1 means predictions span *half* the range of reality — a weighted
average of noisy neighbours regresses to the mean by construction. 4.6–4.96 was
selection bias: it is what the top of any sorted list looks like. Shrinking made
MAE worse on both test sets, exactly as the arithmetic predicts. The correction
is `pred_gain`, which expands.

**3. Removing consensus from the CF input did almost nothing.** The theory was
that `z` contains the crowd's opinion, so averaging `z` reconstructs consensus.
Directionally right, quantitatively wrong: consensus explains only **16.3%** of
the variance in `z`, so subtracting it leaves 91.5% of the spread and changes
the ranking barely at all. The decomposition is kept (`use_residual_input`,
`consensus_beta`) because it makes the split legible in the output — every
recommendation now says what the crowd thinks and how far from that you are
predicted to land — but it is a diagnostic, not the fix.

What *did* matter alongside coverage was `rank_shrinkage_k`: discounting the
personal term by the evidence behind it. Lowering `min_support` without it puts
"predicted 5.00 stars, support 0.0 from 1 neighbour" at the top of the list.
Median percentile measures recall and is blind to that; the top of the list is
precision. They are different questions and they pull in opposite directions.

### Your export specifically

| | value |
|---|---|
| films matched to MovieLens | 431 / 605 (71.2%) |
| MovieLens corpus after filtering | 31.7M ratings, 200,948 users, 23,350 films |
| similarity build | 78 s, 4.67M nonzeros |
| predicted-star MAE | 0.908 vs 1.061 for "always guess your mean" |
| non-film entries filtered from results | 585 of 23,350 |

| | Q1 | Q2 | Q3 | Q4 | Q5 |
|---|---|---|---|---|---|
| n | 16 | 17 | 11 | 46 | 217 |
| item-item CF (now) | 0.650 | 0.833 | 0.453 | **0.910** | 0.921 |
| item-item CF (before) | 0.638 | 0.698 | 0.460 | 0.831 | 0.951 |
| by consensus | 0.576 | **0.853** | 0.538 | 0.802 | 0.924 |

**Do not tune on this table.** With n=11 in Q3 the cells are noise, which is
precisely why `synthetic.py` exists. It is here to confirm the pipeline runs end
to end on a real export, not to choose parameters. Note also that the match rate
is *lower* than the 73.3% previously recorded while being *more* correct — see
known problem 6.

## Known problems

1. **Genuine obscurity is still mostly unreachable.** Q1 coverage went 16% → 40%,
   which is a real improvement and still means the engine declines on 3 films in
   5. Films with 20–60 MovieLens ratings cannot accumulate co-rating support, and
   no amount of threshold-tuning invents evidence that is not there. This needs
   the content-based track, not CF.
2. **No "semi-similar" retrieval.** Every mode here varies *popularity*
   (`popularity_alpha`, and the windows in `MODES`). Nothing varies *kind*. There
   is no diversity term and no mechanism for "adjacent but not obvious" — `gem`
   finds obscure films your neighbours liked, not tangentially related ones.
3. **`pred_gain` is fitted globally.** 1.17 comes from MovieLens raters. The
   attenuation is probably user-specific — on the 605-rating export MAE moved
   0.887 → 0.908, partly because the engine now also answers about harder films.
   Fitting the gain per user from their own holdout is a contained piece of work.
4. **The same-work dedup is a heuristic.** It reads part markers out of titles
   ("Part II", "Vol. 1", "Voyna i mir III. - Borogyino") and links translated
   alternates through MovieLens's parentheses. It cannot know that "Fast Five"
   belongs with "2 Fast 2 Furious", because nothing in the title says so.
5. **26% of the library still does not match.** Of the 174 misses: ~64 are post
   2023 (MovieLens ends Oct 2023), ~30 Soviet/Kazakh films absent from the
   catalogue, the rest TV or genuinely missing. MovieLens titles are also
   unreliable — *Endgame* is filed as "Avengers: Infinity War - Part II".
6. **MovieLens links.csv contains wrong IMDb ids.** It has two entries for
   *Enron: The Smartest Guys in the Room*, one pointing at a TV episode. The
   film filter removes the mis-linked duplicate, which is the right outcome by
   luck rather than design.

## Where to pick up

In priority order.

1. **The content-based track (TMDB).** This is now the only thing standing
   between the project and its stated goal. CF is excellent where it has
   evidence and silent where it does not, and the silence is concentrated
   exactly on the obscure films the tool exists to surface. TMDB keywords,
   crew and `similar` would give a signal that does not need co-raters, and
   would also cover the 64 post-2023 films MovieLens cannot see.
2. **Semi-similar retrieval.** Known problem 2. Worth designing deliberately
   rather than falling out of a popularity knob: a diversity penalty over the
   returned set, or retrieval at a controlled similarity *band* rather than a
   maximum.
3. **Per-user `pred_gain`.** Known problem 3. Fit the attenuation on the user's
   own holdout instead of borrowing MovieLens's.
4. **IMDb `title.akas` for matching.** The dump is already being downloaded for
   `titleType`; `akas` is the same source and would attack the ~30 Soviet and
   Kazakh misses directly.
5. **LLM taste profile** from the review text, and the written pitch.

Re-run `scripts.eval_synthetic` after any scoring change, and read the
*rankable* row before the percentile row. `scripts.recommend --eval` on one
export cannot tell you whether a change worked.

## Data sources considered

| Source | Per-user ratings? | Use |
|---|---|---|
| MovieLens 32M | **yes** — the only option | CF substrate, tmdbId mapping |
| MovieLens ml-latest | yes; 330k users but catalogue ends Jul 2023 and it is a shifting dev dataset | possible density upgrade |
| IMDb non-commercial | **no** — aggregate ratings only | `titleType` for the film/TV filter (in use), `title.akas` for matching, crew for a content model |
| TMDB API | no | metadata, keywords, watch providers, behaviour-derived `similar`/`recommendations` for post-2023 films |
| Netflix Prize | yes, but 17,770 films all pre-2005, withdrawn after the de-anonymisation suit | no |
