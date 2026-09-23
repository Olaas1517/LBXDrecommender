"""Central configuration: paths and the tunable knobs of the algorithm.

Every magic number in this project lives here, with a comment explaining what it
does and which direction to move it. Nothing else should hardcode a constant.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# ---------------------------------------------------------------- paths

ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"

ML_DIR = DATA_RAW / "ml-32m"
ML_ZIP = DATA_RAW / "ml-32m.zip"


@dataclass(frozen=True)
class NormalizeConfig:
    """Stage 0: turning raw stars into a comparable taste signal."""

    # Floor on a user's rating standard deviation.
    #
    # WHY: z-scores divide by sigma. Someone who rated 600 films entirely within
    # 3.5-4.0 stars has sigma ~= 0.15, which would inflate ordinary rating noise
    # into z-scores of +/-3 ("this is the greatest film I have ever seen").
    # The floor encodes a prior: we don't believe anyone's true internal scale is
    # tighter than half a star. Raise it to be more conservative about
    # narrow-range raters.
    sigma_floor: float = 0.5

    # Half-life, in years, for recency weighting of the *taste profile*.
    #
    # WHY 3 years: taste drifts, but slowly. A 3-year half-life means a film you
    # rated 3 years ago counts half as much as one from last month, and one from
    # 9 years ago counts an eighth. Set to None to disable decay entirely.
    # NOTE: decay applies to the taste profile only, never to the "already seen"
    # filter -- you have still seen a film you watched in 2011.
    recency_half_life_years: float | None = 3.0

    # Should film likes (likes/films.csv) contribute to the taste signal?
    #
    # WHY default False: likes are a noisy, per-user-inconsistent signal. Many
    # members treat the heart as a bookmark or a social gesture rather than an
    # endorsement. Turn it on and let the eval harness tell you whether it helps
    # on YOUR data -- that is exactly the kind of question the harness exists for.
    use_likes: bool = False

    # If likes are enabled, how much of a z-score bump a like is worth.
    like_bonus_z: float = 0.25


@dataclass(frozen=True)
class MatchConfig:
    """Letterboxd title+year -> MovieLens movieId matching."""

    # How many years of slack when matching titles.
    #
    # WHY 1: Letterboxd and MovieLens sometimes disagree by a year on release
    # date (festival premiere vs general release, or a late-December film). More
    # than 1 year of slack starts matching genuine remakes to each other, which
    # is a much worse error than a miss.
    year_tolerance: int = 1

    # Minimum rapidfuzz score (0-100) to accept a fuzzy title match.
    #
    # WHY 88: below roughly 85 you start pairing sequels and same-franchise
    # entries ("Ocean's Eleven" / "Ocean's Twelve" score ~86). 88 with a year
    # constraint is conservative; misses go to a report file for eyeballing
    # rather than being silently guessed.
    fuzzy_threshold: int = 88

    # Must a candidate's year agree (within year_tolerance) before we accept it?
    #
    # WHY THIS EXISTS: the matcher used to accept a UNIQUE title match with no
    # year check at all, even when the export gave a year and it disagreed
    # wildly. That quietly defeated the remake guard this module is built
    # around -- a 2024 film whose title happens to appear once in MovieLens got
    # bound to an unrelated older film, and nothing in the output said so.
    #
    # Set False to restore the old permissive behaviour. It does buy a handful
    # more matches; they are simply not matches you can trust, and a wrong match
    # is far more damaging than a miss because it feeds a real rating into the
    # wrong film's row.
    require_year_agreement: bool = True


@dataclass(frozen=True)
class CFConfig:
    """Item-item collaborative filtering."""

    # Ignore MovieLens films with fewer than this many ratings.
    #
    # WHY 20: below ~20 ratings a film's rating vector is too sparse for its
    # similarity to anything to be meaningful, and it bloats the matrix. This is
    # the single biggest lever on precompute time and memory. It does cost us
    # coverage of the very deepest obscurities -- an acceptable trade because a
    # film with 8 ratings gives us nothing to reason from anyway.
    min_item_ratings: int = 20

    # Ignore MovieLens users with fewer than this many ratings.
    # WHY 5: users with 1-4 ratings add noise and no co-rating structure.
    min_user_ratings: int = 5

    # Shrinkage constant lambda for similarity confidence.
    #
    # WHY this exists (the most important knob in the project): two films rated
    # by only 3 people in common can show a correlation of 1.0 purely by chance.
    # Uncorrected, these spurious perfect matches dominate the top of every
    # recommendation list and the whole system produces garbage.
    #
    # We multiply each raw similarity by n_ij / (n_ij + lambda), where n_ij is
    # the number of users who rated both films. The shape is what makes it good:
    #   n_ij = lambda      -> factor 0.5  (half-trusted)
    #   n_ij >> lambda     -> factor -> 1 (trusted as-is)
    #   n_ij << lambda     -> factor -> 0 (discarded)
    # So lambda reads directly as "how many co-raters before I half-believe it".
    # This is standard empirical-Bayes shrinkage toward zero.
    shrinkage_lambda: float = 60.0

    # Keep only the top-K most similar neighbours per film.
    #
    # WHY 200: the full 87k x 87k similarity matrix is ~30 GB dense and mostly
    # noise. Truncating to the strongest 200 neighbours per film keeps the signal,
    # makes the matrix sparse enough to hold in RAM, and acts as a second noise
    # filter. Standard practice; 100-500 all work.
    top_k_neighbours: int = 200

    # The novelty dial. Applied at SCORING time, never baked into the matrix.
    #
    # WHY at scoring time: the similarity matrix takes minutes to build and this
    # is the knob you will actually fiddle with, so retuning stays instant.
    #
    #   rank_score = pred_z - popularity_alpha * log10(n_i / median_n)
    #
    # Subtractive in log-popularity space rather than multiplicative, so it stays
    # monotonic and behaves sanely when pred_z is negative. The units are
    # meaningful: it is "how many z-units of predicted rating I will trade away
    # for a film that is 10x less widely seen".
    #   0.0  -> no preference; recommends crowd-pleasers (the documented failure
    #           mode of the popular open-source Letterboxd recommenders)
    #   0.15 -> mild nudge toward the less-seen
    #   0.4  -> aggressive; surfaces real obscurities, and more misses
    popularity_alpha: float = 0.15

    # Minimum total similarity mass before we are willing to make a claim.
    #
    # WHY: a prediction assembled from 2 weak neighbours is a guess wearing a
    # number. Films below this threshold are unrankable rather than silently
    # ranked, so the engine refuses instead of embarrassing itself.
    #
    # MEASURED on 150 held-out MovieLens users x 3 splits. "cov" is the fraction
    # of held-out liked films the engine will rank at all; "med" is the median
    # percentile including the refusals.
    #
    #   min_sup     Q1 cov/med    Q2 cov/med    Q3 cov/med    Q5 med   ALL
    #     0.50      0.00 / .393   0.05 / .448   0.09 / .630    .986    .985
    #     0.10      0.10 / .426   0.21 / .443   0.33 / .702    .972    .971
    #     0.03      0.30 / .482   0.70 / .884   0.62 / .902    .956    .956
    #     0.01      0.55 / .854   0.74 / .862   0.74 / .896    .947    .946
    #     0.00      1.00 / .683   1.00 / .707   1.00 / .837    .736    .743
    #
    # Read the shape of that. Loosening from 0.10 to 0.01 roughly triples how
    # often the engine will answer about an obscure film, and doubles how well
    # it does when it does (Q1 .426 -> .854), for about 2.5% on the popular
    # films nobody needs help finding. Dropping to 0 collapses everything:
    # predictions assembled from no evidence flood the ranking and Q5 falls off
    # a cliff. So the floor is doing real work; it was just set five times too
    # high for a recommender whose point is the mid-tail.
    #
    # 0.03 is the default rather than 0.01 because the Q1 cells are small
    # (n=16-30) and their medians are jumpy, while Q2/Q3 carry more data and are
    # nearly as good at 0.03. `gem` mode drops it further, which is the right
    # place to spend coverage on obscurity.
    min_support: float = 0.03

    # ---------------------------------------------------------------- scoring
    #
    # The three knobs below are the answer to the project's central open problem:
    # "CF ties the consensus baseline, so the personalisation is decorative."
    #
    # WHY IT TIED. The old scorer fed each rated film's z-score into the weighted
    # average. But z contains the consensus inside it: z_ui = (how good the film
    # is) + (how much YOU disagree). Averaging z over a film's neighbours
    # therefore reconstructs "films like the ones you rated highly are good
    # films" -- which is a restatement of consensus. Ranking by it could not help
    # but track the consensus baseline.
    #
    # THE FIX. Subtract the crowd out of the input, and add it back exactly once,
    # under a knob:
    #
    #     d_i        = z_i - zbar_i           your DISAGREEMENT with the crowd
    #     pred_d(j)  = weighted avg of d_i    predicted disagreement
    #     pred_z(j)  = zbar_j + pred_d(j)     honest absolute prediction
    #     rank_score = beta * zbar_j + pred_d(j) - popularity penalty
    #
    # Consensus now enters in exactly one place, with a dial on it, instead of
    # leaking in through the input. This is standard double-centering (Koren
    # 2010): the old code removed the USER baseline (that is what z is) but never
    # removed the ITEM baseline.

    # Centre each film's rating vector before measuring similarity.
    #
    # WHY THIS IS A SEPARATE KNOB FROM use_residual_input: they remove consensus
    # from two different places. use_residual_input changes what is AVERAGED;
    # this changes what "similar" MEANS.
    #
    # Without it, similarity is adjusted cosine over z-vectors, and two
    # unrelated but widely-loved films share a large positive constant
    # component that inflates their cosine toward each other for no reason
    # beyond both being liked. Subtracting each film's own mean first turns the
    # measure into a Pearson correlation over co-raters: "do the people who
    # over-rate this film relative to the crowd also over-rate that one". That
    # is co-deviation, which is a genuinely personal structure, where plain
    # cosine partly measures co-popularity.
    #
    # Costs a full matrix rebuild to change, unlike the scoring knobs.
    item_centered: bool = False

    # Feed disagreement-with-the-crowd into the CF average, rather than raw z.
    #
    # Set False to reproduce the pre-fix behaviour exactly (together with
    # consensus_beta=0.0). Kept switchable because the sweep in scripts.tune
    # compares the two directly, and a claim like "the new scorer is better"
    # should stay re-checkable rather than becoming folklore.
    use_residual_input: bool = True

    # How much of the film's own consensus standing to put back into the RANKING.
    #
    #   1.0 -> rank by predicted absolute enjoyment. Good films rank high.
    #   0.5 -> split the difference.
    #   0.0 -> rank purely by predicted disagreement: "films you will like far
    #          more than the crowd does". Maximally personal, and happy to
    #          recommend a mediocre film you will merely dislike less.
    #
    # Note this affects rank_score ONLY. pred_z stays the honest absolute
    # prediction so that the star numbers remain falsifiable against reality.
    consensus_beta: float = 1.0

    # Shrink the consensus estimate itself for thinly-rated films.
    #
    # WHY: zbar_i for a film with 21 ratings is a noisy estimate of its standing.
    # Left alone, the thinnest films in the catalogue supply the most extreme
    # consensus values and dominate whichever end of the ranking beta favours.
    # Same empirical-Bayes shape as shrinkage_lambda, shrinking toward z=0, which
    # is the global average film by construction.
    consensus_shrinkage_lambda: float = 30.0

    # Support-based shrinkage on the prediction itself.
    #
    # DEFAULT 0.0 -- OFF -- AND THE REASON IS WORTH READING, because the
    # project's own notes asked for the opposite.
    #
    # The stated problem was "predictions are overconfident, everything comes
    # back 4.6-4.96 stars, needs support-based shrinkage". That observation was
    # real but the diagnosis was backwards: 4.6-4.96 is what the TOP of a sorted
    # list looks like. Sorting 23,000 films by predicted rating and reading the
    # first fifteen will show you the highest predictions in the catalogue no
    # matter how well calibrated the model is. It is selection bias, not
    # overconfidence.
    #
    # MEASURED on 8,647 held-out ratings from 150 users:
    #     actual_z = 1.172 * pred_z - 0.046
    #     sd(pred_z) = 0.488   vs   sd(actual_z) = 0.985
    # A slope above 1 means predictions are UNDER-dispersed -- they span half
    # the range of reality, because a weighted average of noisy neighbours
    # regresses toward the mean by construction. And the slope RISES with
    # support (0.945 at support 0.5-1, up to 1.376 above 10), so shrinkage
    # corrects hardest exactly where correction is least warranted.
    #
    # Switching it on made MAE worse on both test sets (0.887 -> 0.981 on the
    # real export, 0.588 -> 0.639 pooled), which is what the arithmetic above
    # predicts. Left here as a knob because it is a reasonable thing to want to
    # try; left at 0 because the data says it is wrong.
    pred_shrinkage_k: float = 0.0

    # Support discount applied to the personal term in the RANKING.
    #
    # This is the knob that makes a low min_support safe, and it is a different
    # thing from pred_shrinkage_k above even though the arithmetic looks alike.
    #
    # THE TRAP IT EXISTS TO AVOID. Lowering min_support to buy coverage of
    # obscure films improves the median percentile of held-out liked films a
    # great deal -- and produces a top-of-list full of rubbish. Both are true at
    # once, because they are different questions. Median percentile measures
    # RECALL: can a film we know is good be ranked highly. The top of the list
    # is PRECISION: is what we are about to show you any good. A film with one
    # weak neighbour can be handed pred_d = +1.96z on no evidence, which does
    # nothing much to recall and puts "Burt's Buzz (2014), predicted 5.00 stars,
    # support 0.0 from 1 neighbour" at the top of your recommendations.
    #
    # So: rank by the personal signal DISCOUNTED by how much evidence stands
    # behind it,
    #     rank_score = beta * zbar_j + pred_d * support/(support + k) - penalty
    # and a thin-evidence film falls back toward the crowd's opinion of it
    # rather than shooting to the top on noise. It can still be ranked -- which
    # is what min_support=0.03 buys -- it just cannot win on nothing.
    #
    # Deliberately NOT applied to pred_z: the star prediction stays an honest
    # statement about the evidence, and the measurement in pred_shrinkage_k
    # shows shrinking it makes it worse.
    rank_shrinkage_k: float = 1.0

    # Gain applied to pred_z before it is converted to stars.
    #
    # This is the correction the measurement above actually calls for: the
    # predictions need EXPANDING, not shrinking. 1.17 is the fitted slope of
    # actual on predicted, i.e. the value that minimises squared error.
    #
    # Applied to the star prediction ONLY, never to rank_score. A positive
    # constant multiplier cannot change a ranking -- it is monotone -- so this
    # can only affect the falsifiable claim ("we think you'll give this 4.2")
    # and never the order films appear in. That separation is the whole reason
    # pred_z and rank_score are different quantities.
    pred_gain: float = 1.17


@dataclass(frozen=True)
class FilterConfig:
    """What is allowed to appear in a recommendation list at all.

    These are presentation-layer filters, deliberately not baked into the
    similarity matrix or the scores. A TV episode still carries real
    collaborative signal -- people who liked one Sherlock episode liked the
    others -- so it should keep contributing to similarity. It just must not be
    offered as a film to go and watch.
    """

    # Drop non-film entries using IMDb's title.basics.titleType.
    #
    # WHY this is the clean fix: MovieLens inherits TV from its contributors
    # ("His Last Vow ()" is a Sherlock episode) and nothing in the MovieLens
    # metadata distinguishes it. IMDb publishes the authoritative type, keyed by
    # the imdbId MovieLens already gives us in links.csv. Free, no API key.
    drop_non_films: bool = True

    # IMDb titleTypes that are NOT films, and so must never be recommended.
    #
    # Stated as a DENYLIST rather than an allowlist, deliberately. An allowlist
    # fails closed: the first version of this said film_title_types =
    # ("movie", "tvMovie"), which quietly deleted 573 short films from the
    # catalogue -- including Wallace & Gromit: The Wrong Trousers (17,652
    # ratings), Paperman, and the 2013 short Whiplash. A short is a film you can
    # sit down and watch, Letterboxd catalogues them, and this library contains
    # Un Chien Andalou. A denylist fails open instead: an unrecognised or newly
    # invented titleType stays in the catalogue rather than silently vanishing.
    #
    # The problem being solved is specifically TV -- "His Last Vow ()" is a
    # Sherlock episode -- so the list names television and nothing else.
    drop_title_types: tuple[str, ...] = (
        "tvEpisode", "tvSeries", "tvMiniSeries", "tvSpecial", "tvShort", "tvPilot",
    )

    # Are short films recommendable?
    #
    # Genuinely a matter of taste rather than correctness, so it is a switch
    # rather than a decision made on your behalf. Letterboxd catalogues shorts
    # and plenty of people log them -- this library contains Un Chien Andalou --
    # but "you have 90 minutes free, what should you watch" is a different
    # question from "what is good", and a list where half the entries run 11
    # minutes is not answering it.
    #
    # MovieLens holds 573 shorts, including Wallace & Gromit: The Wrong Trousers
    # (17,652 ratings) and the 2013 short Whiplash.
    include_shorts: bool = True

    def excluded_types(self) -> tuple[str, ...]:
        """The full denylist, once the shorts switch is applied."""
        if self.include_shorts:
            return self.drop_title_types
        return self.drop_title_types + ("short",)

    # Collapse multiple entries of the same work down to one recommendation.
    #
    # WHY: rating one part of Bondarchuk's War and Peace returns the other three
    # parts as the top three recommendations. They are technically correct and
    # completely useless -- the user has already decided about that work. Same
    # for a film indexed under both its theatrical and extended cuts.
    dedupe_same_work: bool = True

    # Never show a recommendation that cannot be explained.
    #
    # The engine's whole claim over matrix factorisation is that every result
    # decomposes into "because you rated X, Y and Z the way you did". A result
    # assembled from a single neighbour has no such story -- it prints with no
    # `because` line at all -- and in gem mode, where the support floor is
    # deliberately low, those were reaching the top of the list.
    #
    # This is a presentation rule, not a scoring one: the film stays ranked and
    # still counts in evaluation, it just is not offered as advice. Two is the
    # minimum that can honestly be called a pattern.
    min_neighbours_to_show: int = 2

    # Also suppress works the member has already seen ANY part of.
    #
    # WHY: excluding "films you have watched" is done by film identity, so
    # watching War and Peace Part II removes exactly that entry and leaves
    # Parts I, III and IV free to be recommended back. They are the same work.
    # Someone who has started a four-part Bondarchuk adaptation does not need a
    # recommender to suggest the rest of it.
    dedupe_against_seen: bool = True

    # How many entries from one franchise/series may appear in a single list.
    #
    # WHY not 1: a series is not the same thing as a work. Recommending two
    # Kieslowski Three Colours films is a reasonable list; recommending all
    # eight Fast & Furious films is not. 2 keeps a little room and still stops a
    # list being eaten by one franchise. Set to 0 to disable.
    max_per_series: int = 2


@dataclass(frozen=True)
class EvalConfig:
    """Offline evaluation."""

    # Fraction of the user's ratings held out as the test set.
    holdout_frac: float = 0.2

    # A held-out film counts as "relevant" (something we should have found) if
    # the user rated it at least this many sigma above their own mean.
    #
    # WHY 0.5: we are not trying to predict everything they watched, we are
    # trying to predict what they LIKED. Half a sigma above their own average is
    # a defensible line for "this was a good watch for me".
    relevant_z_threshold: float = 0.5

    ks: tuple[int, ...] = (10, 20, 50, 100)

    # Random seed so that tuning runs are comparable to each other.
    seed: int = 20260730


NORMALIZE = NormalizeConfig()
MATCH = MatchConfig()
CF = CFConfig()
FILTER = FilterConfig()
EVAL = EvalConfig()
