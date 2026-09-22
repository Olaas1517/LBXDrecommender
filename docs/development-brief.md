# Development brief — 2026-09-22

How the scoring rework happened, including the parts that went wrong. Written
because the *wrong* turns are the useful record: three of the project's prior
diagnoses were confidently stated and measurably false, and the only reason that
surfaced is that each one got measured instead of implemented on faith.

## Starting point

A complete, well-documented scaffold — ingest → normalize → match → CF → eval —
with an honest README naming its own central failure: item-item CF only *tied* a
"recommend acclaimed films to everybody" baseline. The README proposed a fix
(score on the consensus residual) and listed five more items.

Nothing was broken. The question was whether the personalisation did anything.

## The first hypothesis, and why it was wrong

The README said scoring "ignores `residual`". The sharper version seemed to be:
`z` already *contains* consensus, since `z = (how good the film is) + (how much
you disagree)`. Averaging `z` over a film's neighbours therefore reconstructs
"films like the ones you rated highly are good films" — a restatement of
consensus. So it could never beat the baseline.

That reasoning is sound and the conclusion was still wrong. Implemented as
double-centering (subtract the crowd out of the input, add it back once under a
`consensus_beta` dial), it moved Q2 from 0.698 to 0.699. Nothing.

The diagnostic that explained it took two minutes and should have come first:

```
consensus explains 16.3% of the variance in z
subtracting it leaves 91.5% of the original spread
```

Directionally right, quantitatively irrelevant. The decomposition survives in
the code because it makes the split legible in the output — every recommendation
now states the crowd's position and your predicted distance from it — but it is
a diagnostic, not the fix.

**Lesson:** a correct-sounding causal story deserves a variance budget before it
deserves an implementation.

## The measurement was broken, which was the real finding

The stratified table — the README's "comparison that matters" — had a hole.
Films below `min_support` are all assigned `rank_score = -inf`. They therefore
tie, and a stable argsort orders ties by index. Their "percentile" was a
position in a tie-break, not a measurement, and the table averaged those in with
real rankings.

It showed up as a pattern that cannot happen in a working experiment: Q1, Q2 and
Q3 were **identical to three decimals across every scoring configuration**.
Scoring changes that produce bit-identical results are not subtle; they mean the
cells are not measuring scoring.

Splitting "how well does it rank?" from "will it rank this at all?" changed the
picture completely:

| | Q1 | Q2 | Q3 | Q4 | Q5 |
|---|---|---|---|---|---|
| fraction rankable (before) | 0.158 | 0.200 | 0.329 | 0.675 | 0.980 |
| median percentile *among rankable* | 0.976 | 0.997 | 0.987 | 0.983 | 0.973 |

The engine was never bad at ranking. It was excellent, and silent. The problem
was coverage, and the tie-break artefact had been hiding that behind numbers
that looked like mediocre ranking quality.

**Lesson:** when a metric can be computed for an item the system refused to
score, it will be, and the result will look like a result.

## The second wrong diagnosis

README known-problem #1: *"Predictions are overconfident. Everything comes back
4.6–4.96 stars. Needs support-based shrinkage on `pred_z`."*

Implemented as asked; MAE got worse on both test sets. Calibration, fitted on
8,647 held-out ratings:

```
actual_z = 1.172 * pred_z - 0.046
sd(pred_z) = 0.488   vs   sd(actual_z) = 0.985
```

Slope above 1 means predictions span *half* the range of reality. A weighted
average of noisy neighbours regresses to the mean by construction — it cannot
be overconfident. And the slope *rises* with support (0.945 → 1.376), so
shrinkage corrects hardest exactly where correction is least warranted.

The original observation was real; the inference was not. 4.6–4.96 is what the
top of *any* sorted list looks like. It was selection bias. The correction is
`pred_gain`, which expands, and it is applied to the star prediction only —
being a positive constant it cannot reorder anything.

**Lesson:** "the top of my ranked list all has high scores" is not evidence
about calibration.

## Recall and precision pulled in opposite directions

With coverage identified as the bottleneck, lowering `min_support` from 0.1 to
0.01 roughly tripled how often the engine would answer about an obscure film,
and doubled Q1 median percentile. Clear win on every number in the sweep.

Then `--mode gem` produced this:

```
Burt's Buzz (2014)
  predicted 5.00 stars | 28 MovieLens ratings | support 0.0 from 1 neighbours
```

Every entry pinned at 5.00 stars on a single weak neighbour. Median percentile
of held-out liked films measures **recall** — can a known-good film be ranked
highly — and is completely blind to garbage arriving at the top, which is
**precision**, and is the only thing a user actually reads.

The resolution was not a threshold but `rank_shrinkage_k`: discount the personal
term by the evidence behind it, so a thin-evidence film falls back toward the
crowd's opinion instead of winning on noise. It can still be *ranked* — which is
what the lower floor buys — it just cannot *win*. Settled at `min_support=0.03`
(0.01 in gem mode, where obscurity is the point).

**Lesson:** optimising a recall metric will happily destroy precision, silently.

## Bugs found by running against a real export

Synthetic fixtures found none of these. All three needed real data.

1. **`ingest.py` — deleted entries could become your taste profile.** Real
   exports ship `deleted/diary.csv` and `orphaned/reviews.csv` beside the real
   ones. Both end in `/diary.csv`, and the reader accepted an exact *or* suffix
   match in a single pass, so it took whichever the ZIP happened to list first.
   It worked by luck. Exact match now wins outright; the suffix fallback skips
   those two directories.

2. **`match.py` — the remake guard had a hole.** A title appearing exactly once
   in MovieLens was accepted *regardless of year*, under the label "exact title,
   no year check". On a 605-film library that bound Napoleon (2023) → Napoléon
   (1927), The Killer (2023) → the 1989 John Woo film, Wicked (2024) → Wicked
   (1998), and six more — feeding real ratings into the wrong films' rows.

   The first fix was too blunt and cost 14 matches. Inspecting them split the
   group cleanly: 9 genuinely wrong, 5 correct and lost because *MovieLens has
   no year for them at all*. The original comment said "no year to check
   against" but the code checked whether the **export** had a year, when what
   matters is whether the **candidate** does. Nothing can disagree with a year
   that does not exist.

3. **`cf.py` — `shrinkage_lambda=0` produced NaNs.** `cnt/(cnt+0)` is `0/0` for
   any pair with no co-raters, and `NaN != 0` is `True`, so the NaNs passed the
   filter straight into the similarity matrix. Only triggered by sweeping lambda
   to zero — which is a reasonable thing to want to do.

## Tests

83 tests, synthetic fixtures, no downloads — a suite that needs a 227 MB corpus
is a suite nobody runs. Two of them were written wrong first, and both failures
were informative:

- A test asserting residual scoring beats legacy at cluster separation, on a
  fixture where consensus is ~0 by construction. There was nothing to remove.
  That claim is empirical and belongs in the sweep, not a unit test; it was
  replaced with a test of the *dial's semantics*, which is unit-testable.
- A shrinkage test whose three raters rated everything identically — giving them
  σ=0, z=0, and a fixture that proved nothing. They now rate a third film
  differently.

The useful ones encode the failure they prevent: `deleted/` entries shadowing
real ones (with archive order deliberately hostile), a unique title with a wrong
year, thin evidence winning a ranking, `to_stars` staying an exact inverse while
`predicted_stars` expands.

## Validation strategy

One export yields n=11 in the Q3 cell. Medians over eleven observations move
0.2 on noise, and every knob in the project had been chosen that way.

`synthetic.py` treats held-out MovieLens users as stand-in members: 300 users,
20,545 observations, and — importantly — excluded from the similarity build by
`build_cf`, so it is not testing the system against its own training data.
`verify_disjoint` refuses to run otherwise rather than quietly reporting a
number measured on training data.

It cannot validate everything. MovieLens users are not Letterboxd users, and the
harness says so in its docstring: it measures the *scoring*, which is
population-general, not the *matching* or catalogue coverage, which are specific
to one person's library.

## Result

Median percentile of held-out liked films, 300 held-out users:

| | Q1 | Q2 | Q3 | Q4 |
|---|---|---|---|---|
| CF now | 0.542 | 0.809 | 0.874 | 0.947 |
| CF before | 0.469 | 0.467 | 0.670 | 0.957 |
| by consensus | 0.445 | 0.697 | 0.661 | 0.784 |

The old configuration *lost* to consensus on Q2. It now wins on every stratum
except Q5, where popularity is unbeatable by construction.

Leave-one-out on the owner's own library: Battle of Algiers ranks 48 of 22,327,
Dr. Strangelove 120, Apocalypse Now 313. A film loved (≥4.5) outranks a film
hated (≤1.0) 83% of the time.

Note that hated films still land in the top ~16% in absolute terms. That is less
alarming than it first looks: they are not random films, they are films this
person *chose to watch*, drawn from the same neighbourhood as the ones they
love. Ranking them above 84% of the catalogue is defensible. Whether it is
desirable is a product question, and the honest number to quote is the 0.83
separation, not the absolute position.
