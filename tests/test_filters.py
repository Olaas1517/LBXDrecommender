"""Presentation filters: TV, shorts, and same-work collapsing."""

from __future__ import annotations

import pandas as pd
import pytest

from lbxd.config import FILTER, FilterConfig
from lbxd.filters import diversify, series_key, work_key


class TestWorkKey:
    @pytest.mark.parametrize("a,ya,b,yb", [
        ("War and Peace, Part I: Andrei Bolkonsky", 1966,
         "War and Peace, Part II: Natasha Rostova", 1966),
        # The form that actually shipped a duplicate to the user: the part
        # marker sits mid-title, followed by that part's own subtitle.
        ("Voyna i mir III. - Borogyino", 1967,
         "Voyna i mir IV. - Pierre Bezukhov", 1967),
        ("Kill Bill: Vol. 1", 2003, "Kill Bill: Vol. 2", 2004),
        ("Nymphomaniac: Vol. I", 2013, "Nymphomaniac: Vol. II", 2013),
    ])
    def test_parts_of_one_work_share_a_key(self, a, ya, b, yb):
        assert work_key(a, ya) == work_key(b, yb)

    @pytest.mark.parametrize("title,year", [
        ("Apollo 13", 1995), ("Ocean's Eleven", 2001), ("Se7en", 1995),
        ("Mississippi Burning", 1988), ("X", 2022), ("2001: A Space Odyssey", 1968),
    ])
    def test_ordinary_titles_are_not_mangled(self, title, year):
        """A false merge silently deletes a legitimate recommendation, so the
        stripping must never fire on a title that merely looks numeric."""
        assert work_key(title, year).startswith("title::")
        assert len(work_key(title, year).split("::")[1]) >= 1

    def test_remakes_stay_distinct(self):
        assert work_key("Ambulance", 2005) != work_key("Ambulance", 2022)

    def test_a_title_stripped_to_nothing_falls_back(self):
        """'X' must not become an empty key that collides with everything."""
        assert work_key("X", 2022) != work_key("V", 2021)

    def test_alternate_cuts_collapse(self):
        assert work_key("Apocalypse Now (Redux)", 1979) == work_key("Apocalypse Now", 1979)


class TestSeriesKey:
    def test_colon_titles_group(self):
        assert series_key("Star Wars: Episode IV") == series_key("Star Wars: Episode V")
        assert series_key("Three Colors: Blue") == series_key("Three Colors: Red")

    def test_no_colon_means_no_series(self):
        assert series_key("Stalker") is None

    def test_a_tiny_head_is_too_generic_to_group_on(self):
        assert series_key("It: Chapter Two") is None


class TestDiversify:
    def _frame(self, rows):
        return pd.DataFrame(rows, columns=["clean_title", "year", "rank_score"])

    def test_one_entry_per_work_survives(self):
        df = self._frame([
            ("War and Peace, Part I", 1966, 9.0),
            ("War and Peace, Part II", 1966, 8.9),
            ("War and Peace, Part III", 1967, 8.8),
            ("Stalker", 1979, 8.0),
        ])
        out = diversify(df, 10, FILTER)
        assert len(out) == 2
        assert "Stalker" in set(out["clean_title"])

    def test_the_best_ranked_member_is_the_one_kept(self):
        df = self._frame([
            ("Kill Bill: Vol. 1", 2003, 9.0),
            ("Kill Bill: Vol. 2", 2004, 8.0),
        ])
        out = diversify(df, 10, FILTER)
        assert out["clean_title"].iloc[0] == "Kill Bill: Vol. 1"

    def test_a_franchise_cannot_eat_the_whole_list(self):
        df = self._frame([(f"Fast Saga: Part {i}", 2000 + i, 9.0 - i) for i in range(6)]
                         + [("Stalker", 1979, 1.0)])
        out = diversify(df, 10, replace_cfg := FilterConfig(max_per_series=2))
        assert "Stalker" in set(out["clean_title"])

    def test_rank_order_is_preserved(self):
        df = self._frame([("A", 1990, 9.0), ("B", 1991, 8.0), ("C", 1992, 7.0)])
        out = diversify(df, 3, FILTER)
        assert list(out["clean_title"]) == ["A", "B", "C"]

    def test_filters_can_be_switched_off(self):
        df = self._frame([
            ("War and Peace, Part I", 1966, 9.0),
            ("War and Peace, Part II", 1966, 8.9),
        ])
        out = diversify(df, 10, FilterConfig(dedupe_same_work=False, max_per_series=0))
        assert len(out) == 2


class TestShortsSwitch:
    def test_shorts_are_recommendable_by_default(self):
        assert "short" not in FilterConfig().excluded_types()

    def test_the_switch_excludes_them(self):
        assert "short" in FilterConfig(include_shorts=False).excluded_types()

    def test_television_is_always_excluded(self):
        for cfg in (FilterConfig(), FilterConfig(include_shorts=False)):
            for t in ("tvEpisode", "tvSeries", "tvMiniSeries", "tvSpecial"):
                assert t in cfg.excluded_types()


class TestSeenWorkSuppression:
    def test_other_parts_of_a_started_work_are_suppressed(self):
        """Watching War and Peace Part II removes that entry from the candidate
        pool by identity -- and leaves Parts I, III and IV recommendable. They
        are the same work."""
        df = pd.DataFrame(
            [("War and Peace, Part III", 1967, 9.0), ("Stalker", 1979, 8.0)],
            columns=["clean_title", "year", "rank_score"],
        )
        from lbxd.filters import base_titles
        started = base_titles("War and Peace, Part II")
        out = diversify(df, 10, FILTER, exclude_works=started)
        assert list(out["clean_title"]) == ["Stalker"]

    def test_unrelated_films_are_untouched(self):
        df = pd.DataFrame(
            [("Stalker", 1979, 9.0)], columns=["clean_title", "year", "rank_score"]
        )
        from lbxd.filters import base_titles
        out = diversify(df, 10, FILTER, exclude_works=base_titles("Kill Bill: Vol. 1"))
        assert len(out) == 1


    def test_a_translated_alternate_title_still_links_the_parts(self):
        """MovieLens writes the whole work as "War and Peace (Voyna i mir)" and
        its parts as "Voyna i mir III". The link lives in the parentheses."""
        from lbxd.filters import base_titles

        started = base_titles("War and Peace (Voyna i mir)")
        df = pd.DataFrame(
            [("Voyna i mir III. - Borogyino", 1967, 9.0), ("Stalker", 1979, 8.0)],
            columns=["clean_title", "year", "rank_score"],
        )
        out = diversify(df, 10, FILTER, exclude_works=started)
        assert list(out["clean_title"]) == ["Stalker"]

    def test_suppression_only_fires_on_multi_part_entries(self):
        """Having seen "Stalker" must not suppress an unrelated film that merely
        shares a word -- only genuine part-of-a-work entries are affected."""
        from lbxd.filters import base_titles

        df = pd.DataFrame(
            [("Stalker Country", 1999, 9.0)], columns=["clean_title", "year", "rank_score"]
        )
        out = diversify(df, 10, FILTER, exclude_works=base_titles("Stalker"))
        assert len(out) == 1
