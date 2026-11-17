"""
tests/test_un_votes.py
------------------------
UN symbol normalization and the voting-record queries, against a
throwaway SQLite database filled with made-up rows.
"""

import sqlite3
from contextlib import closing

import pytest

from services.un_votes import (
    create_schema,
    get_resolution,
    get_votes,
    normalize_un_symbol,
    search_resolutions,
)


@pytest.mark.parametrize(
    "text, expected",
    [
        ("résolution 77-276", "A/RES/77/276"),
        ("Résolution 77/276", "A/RES/77/276"),
        ("la resolution no. 77/276 de l'Assemblée", "A/RES/77/276"),
        ("A/RES/77/276", "A/RES/77/276"),
        ("a/res/77/276", "A/RES/77/276"),
        ("A/80/L65", "A/80/L.65"),
        ("A/80/L.65", "A/80/L.65"),
        ("A / 80 / L. 65", "A/80/L.65"),
        ("A/80/L.65/Rev.1", "A/80/L.65/Rev.1"),
        ("S/RES/2334 (2016)", "S/RES/2334 (2016)"),
        ("S/RES/2334(2016)", "S/RES/2334 (2016)"),
        ("s/res/2334", "S/RES/2334"),
    ],
)
def test_normalize_un_symbol(text, expected):
    assert normalize_un_symbol(text) == expected


@pytest.mark.parametrize("text", ["", "la résolution 2334", "GA/12760", "A/80/PV.83", "rapport annuel 2025"])
def test_unrecognised_text_gives_none(text):
    assert normalize_un_symbol(text) is None


@pytest.fixture
def votes_db(tmp_path):
    path = tmp_path / "votes.sqlite"
    with closing(sqlite3.connect(path)) as conn, conn:
        create_schema(conn)
        conn.executemany(
            "INSERT INTO resolutions VALUES (?, ?, ?, ?)",
            [
                ("A/RES/77/276", "2023-03-29", "Request for an advisory opinion on climate change", 1),
                ("A/80/L.65", "2026-05-20", "Advisory opinion on climate obligations", 0),
                ("A/RES/80/1", "2025-09-10", "100% renewable energy (test row)", 0),
            ],
        )
        conn.executemany(
            "INSERT INTO votes VALUES (?, ?, ?)",
            [
                ("A/80/L.65", "VALMONT", "Y"),
                ("A/80/L.65", "ALPHA", "N"),
                ("A/80/L.65", "BETA", "A"),
                ("A/80/L.65", "GAMMA", ""),
            ],
        )
    return path


def test_get_resolution_accepts_loose_symbols(votes_db):
    res = get_resolution("résolution 77-276", db_path=votes_db)
    assert res["un_symbol"] == "A/RES/77/276"
    assert res["adopted_without_vote"] is True
    assert get_resolution("A/80/L65", db_path=votes_db)["adopted_without_vote"] is False
    assert get_resolution("A/RES/1/1", db_path=votes_db) is None


def test_get_votes_all_and_by_country(votes_db):
    all_votes = get_votes("A/80/L.65", db_path=votes_db)
    assert {v["country"]: v["vote"] for v in all_votes} == {"VALMONT": "Y", "ALPHA": "N", "BETA": "A", "GAMMA": ""}

    some = get_votes("A/80/L65", countries=["VALMONT", "BETA"], db_path=votes_db)
    assert some == [{"country": "BETA", "vote": "A"}, {"country": "VALMONT", "vote": "Y"}]
    assert get_votes("A/RES/77/276", db_path=votes_db) == []


def test_search_resolutions_by_keyword(votes_db):
    found = search_resolutions("climate", db_path=votes_db)
    assert [r["un_symbol"] for r in found] == ["A/80/L.65", "A/RES/77/276"]  # newest first
    assert [r["un_symbol"] for r in search_resolutions("100%", db_path=votes_db)] == ["A/RES/80/1"]


def test_queries_are_parameterized(votes_db):
    # Treated as plain text: matches nothing and drops nothing.
    assert search_resolutions("x' OR '1'='1", db_path=votes_db) == []
    assert get_votes("A/80/L.65", countries=["VALMONT' OR '1'='1"], db_path=votes_db) == []
    with closing(sqlite3.connect(votes_db)) as conn:
        assert conn.execute("SELECT COUNT(*) FROM votes").fetchone()[0] == 4


def test_vote_values_are_constrained(votes_db):
    with closing(sqlite3.connect(votes_db)) as conn, pytest.raises(sqlite3.IntegrityError):
        conn.execute("INSERT INTO votes VALUES ('A/80/L.65', 'DELTA', 'MAYBE')")
