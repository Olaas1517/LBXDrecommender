"""A Letterboxd-export-driven film recommender.

Design constraints that shape everything in this package:

1. No scraping. Letterboxd's terms of use (section 6.11, added December 2025)
   prohibit automated data collection from their site. Our only inputs are the
   user's own export ZIP (which Letterboxd provides deliberately) and, optionally,
   a member's public RSS feed -- which Letterboxd's own API page points members to
   as the machine-readable route.

2. No Letterboxd API. Their access policy explicitly excludes
   "data-analysis, visualization or recommendation projects" and "LLM or
   GPT-related use" and "private or personal projects". We are all four.

3. Therefore MovieLens is the only collaborative-filtering substrate available,
   and TMDB is the only metadata source. Both are used within their terms.
"""

__version__ = "0.1.0"
