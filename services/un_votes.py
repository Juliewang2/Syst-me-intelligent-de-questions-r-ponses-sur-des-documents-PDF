"""
services/un_votes.py
----------------------
UN General Assembly voting records, kept in a small SQLite database
(data/un_votes.sqlite, built by scripts/load_un_votes.py) and queried
here.

Who voted how is structured data: "how did country X vote on A/RES/77/276"
has one exact answer in a table, whereas retrieving it from PDF text is
error-prone (vote lists are long, wrapped and split across pages). So
votes live in a database and are looked up by UN document symbol.

People write symbols loosely ("résolution 77-276", "A/80/L65"), so
`normalize_un_symbol` maps those spellings to the official form before
any lookup. Every query is parameterized - user text is never pasted
into SQL.

Schema:
    resolutions(un_symbol PK, date, title, adopted_without_vote BOOLEAN)
    votes(un_symbol, country, vote CHECK(vote IN ('Y','N','A','')))
where vote '' = present but not voting / absent.
"""

from __future__ import annotations

import re
import sqlite3
import unicodedata
from contextlib import closing
from pathlib import Path
from typing import List, Optional

from config import get_settings

settings = get_settings()

SCHEMA = """
CREATE TABLE IF NOT EXISTS resolutions (
    un_symbol TEXT PRIMARY KEY,
    date TEXT,
    title TEXT,
    adopted_without_vote BOOLEAN
);
CREATE TABLE IF NOT EXISTS votes (
    un_symbol TEXT NOT NULL REFERENCES resolutions(un_symbol),
    country TEXT NOT NULL,
    vote TEXT NOT NULL CHECK (vote IN ('Y', 'N', 'A', '')),
    PRIMARY KEY (un_symbol, country)
);
CREATE INDEX IF NOT EXISTS idx_votes_country ON votes(country);
"""


# ---------------------------------------------------------------------------
# Symbol normalization
# ---------------------------------------------------------------------------

# Each pattern runs on accent-free, upper-case text; first match wins.
_GA_RESOLUTION = re.compile(r"\bA\s*/\s*RES\s*/\s*(\d{1,3})\s*/\s*(\d{1,4})\b")
_SC_RESOLUTION = re.compile(r"\bS\s*/\s*RES\s*/\s*(\d{1,4})(?:\s*\(\s*(\d{4})\s*\))?")
_GA_DRAFT = re.compile(r"\bA\s*/\s*(\d{1,3})\s*/\s*L\s*\.?\s*(\d{1,4})(?:\s*/\s*REV\s*\.?\s*(\d{1,2}))?\b")
# "résolution 77/276", "resolution 77-276", "res. 77/276" -> GA resolution.
_LOOSE_RESOLUTION = re.compile(r"\bRES(?:OLUTION)?\.?\s*(?:NO\.?\s*|N°\s*)?(\d{1,3})\s*[-/]\s*(\d{1,4})\b")


def _plain_upper(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(c for c in decomposed if not unicodedata.combining(c)).upper()


def normalize_un_symbol(text: str) -> Optional[str]:
    """Official UN symbol found in `text`, or None.

    "résolution 77-276" -> "A/RES/77/276"
    "A/80/L65"          -> "A/80/L.65"
    "S/RES/2334(2016)"  -> "S/RES/2334 (2016)"
    """
    plain = _plain_upper(text or "")
    if m := _GA_RESOLUTION.search(plain):
        return f"A/RES/{int(m[1])}/{int(m[2])}"
    if m := _SC_RESOLUTION.search(plain):
        return f"S/RES/{int(m[1])}" + (f" ({m[2]})" if m[2] else "")
    if m := _GA_DRAFT.search(plain):
        return f"A/{int(m[1])}/L.{int(m[2])}" + (f"/Rev.{int(m[3])}" if m[3] else "")
    if m := _LOOSE_RESOLUTION.search(plain):
        return f"A/RES/{int(m[1])}/{int(m[2])}"
    return None


# ---------------------------------------------------------------------------
# Database access
# ---------------------------------------------------------------------------

def connect(db_path: Optional[Path] = None) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path or settings.un_votes_db_path))
    conn.row_factory = sqlite3.Row
    return conn


def create_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)


def _resolution_dict(row: sqlite3.Row) -> dict:
    data = dict(row)
    if data.get("adopted_without_vote") is not None:
        data["adopted_without_vote"] = bool(data["adopted_without_vote"])
    return data


def get_resolution(un_symbol: str, db_path: Optional[Path] = None) -> Optional[dict]:
    """The resolution's date, title and adoption mode, or None."""
    symbol = normalize_un_symbol(un_symbol) or un_symbol
    with closing(connect(db_path)) as conn:
        row = conn.execute("SELECT * FROM resolutions WHERE un_symbol = ?", (symbol,)).fetchone()
    return _resolution_dict(row) if row else None


def get_votes(
    un_symbol: str, countries: Optional[List[str]] = None, db_path: Optional[Path] = None
) -> List[dict]:
    """[{country, vote}] for a resolution, optionally only some countries."""
    symbol = normalize_un_symbol(un_symbol) or un_symbol
    sql = "SELECT country, vote FROM votes WHERE un_symbol = ?"
    params: list = [symbol]
    if countries:
        sql += f" AND country IN ({', '.join('?' for _ in countries)})"
        params += list(countries)
    with closing(connect(db_path)) as conn:
        return [dict(r) for r in conn.execute(sql + " ORDER BY country", params)]


def _escape_like(text: str) -> str:
    return text.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def search_resolutions(keyword: str, db_path: Optional[Path] = None) -> List[dict]:
    """Resolutions whose title or symbol contains `keyword` (case-insensitive)."""
    pattern = f"%{_escape_like(keyword)}%"
    with closing(connect(db_path)) as conn:
        rows = conn.execute(
            "SELECT * FROM resolutions WHERE title LIKE ? ESCAPE '\\' OR un_symbol LIKE ? ESCAPE '\\' "
            "ORDER BY date DESC",
            (pattern, pattern),
        ).fetchall()
    return [_resolution_dict(r) for r in rows]
