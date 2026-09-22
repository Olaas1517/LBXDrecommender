"""Reading a real export, including the traps a real export contains."""

from __future__ import annotations

import io
import zipfile

import pandas as pd
import pytest

from lbxd.ingest import load_export

RATINGS = "Date,Name,Year,Letterboxd URI,Rating\n2024-01-05,Stalker,1979,https://boxd.it/a,4.5\n2024-02-01,Heat,1995,https://boxd.it/b,3.0\n"
DIARY_REAL = "Date,Name,Year,Letterboxd URI,Rating,Rewatch,Tags,Watched Date\n2024-02-01,Heat,1995,https://boxd.it/b,3.0,Yes,,2024-01-30\n2024-03-01,Solaris,1972,https://boxd.it/c,4.0,No,,2024-02-28\n"
DIARY_DELETED = "Date,Name,Year,Letterboxd URI,Rating,Rewatch,Tags,Watched Date\n2019-01-01,Deleted Film,1999,https://boxd.it/x,1.0,No,,2019-01-01\n"
WATCHED = "Date,Name,Year,Letterboxd URI\n2024-01-05,Stalker,1979,https://boxd.it/a\n2020-06-01,Unrated Classic,1960,https://boxd.it/d\n"
LIKES = "Date,Name,Year,Letterboxd URI\n2024-01-06,Stalker,1979,https://boxd.it/a\n"


def _zip(tmp_path, entries: dict[str, str]):
    p = tmp_path / "export.zip"
    with zipfile.ZipFile(p, "w") as zf:
        for name, body in entries.items():
            zf.writestr(name, body)
    return p


def test_reads_a_flat_export(tmp_path):
    p = _zip(tmp_path, {
        "ratings.csv": RATINGS, "diary.csv": DIARY_REAL,
        "watched.csv": WATCHED, "likes/films.csv": LIKES,
    })
    e = load_export(p)
    assert len(e.ratings) == 2
    assert len(e.likes) == 1
    assert set(e.ratings["title"]) == {"Stalker", "Heat"}


def test_deleted_and_orphaned_copies_never_shadow_the_real_files(tmp_path):
    """Real Letterboxd exports ship deleted/diary.csv and orphaned/reviews.csv.

    Both end with "/diary.csv", so a suffix match that does not prefer the exact
    path can pick whichever the archive happens to list first -- and silently
    build your taste profile out of entries you deleted. Archive order is put
    deliberately wrong here so that a regression cannot hide behind luck.
    """
    p = _zip(tmp_path, {
        "deleted/diary.csv": DIARY_DELETED,      # listed FIRST on purpose
        "orphaned/diary.csv": DIARY_DELETED,
        "ratings.csv": RATINGS,
        "diary.csv": DIARY_REAL,
        "watched.csv": WATCHED,
    })
    e = load_export(p)
    assert len(e.diary) == 2
    assert "Deleted Film" not in set(e.diary["title"])


def test_export_nested_in_a_wrapper_folder_still_loads(tmp_path):
    p = _zip(tmp_path, {
        "letterboxd-someone-2026/ratings.csv": RATINGS,
        "letterboxd-someone-2026/watched.csv": WATCHED,
    })
    e = load_export(p)
    assert len(e.ratings) == 2


def test_directory_exports_work_too(tmp_path):
    d = tmp_path / "export"
    (d / "likes").mkdir(parents=True)
    (d / "ratings.csv").write_text(RATINGS, encoding="utf-8")
    (d / "watched.csv").write_text(WATCHED, encoding="utf-8")
    (d / "likes" / "films.csv").write_text(LIKES, encoding="utf-8")
    e = load_export(d)
    assert len(e.ratings) == 2 and len(e.likes) == 1


def test_watched_titles_is_the_union_not_just_ratings(tmp_path):
    """The already-seen filter must cover films logged but never rated.

    Recommending someone a film they watched in 2020 and did not bother to rate
    is the most visible failure this system can have.
    """
    p = _zip(tmp_path, {
        "ratings.csv": RATINGS, "diary.csv": DIARY_REAL, "watched.csv": WATCHED,
    })
    e = load_export(p)
    seen = set(e.watched_titles()["title"])
    assert "Unrated Classic" in seen, "logged-but-unrated films must still be excluded"
    assert "Solaris" in seen, "diary-only films must still be excluded"
    assert len(seen) > len(e.ratings)


def test_missing_optional_files_are_tolerated(tmp_path):
    p = _zip(tmp_path, {"ratings.csv": RATINGS})
    e = load_export(p)
    assert e.ratings.shape[0] == 2
    assert e.likes.empty and e.watchlist.empty
    assert "likes/films.csv" in e.missing


def test_an_export_without_ratings_fails_loudly(tmp_path):
    p = _zip(tmp_path, {"watched.csv": WATCHED})
    with pytest.raises(ValueError, match="ratings"):
        load_export(p)


def test_half_star_ratings_and_dates_are_parsed(tmp_path):
    p = _zip(tmp_path, {"ratings.csv": RATINGS})
    e = load_export(p)
    assert e.ratings["rating"].tolist() == [4.5, 3.0]
    assert pd.api.types.is_datetime64_any_dtype(e.ratings["logged_date"])
    assert e.ratings["year"].tolist() == [1979, 1995]
