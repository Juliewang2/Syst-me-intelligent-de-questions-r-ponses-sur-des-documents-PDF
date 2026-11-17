"""
scripts/load_un_votes.py
--------------------------
Loads a voting-data CSV exported from the UN Digital Library into
data/un_votes.sqlite (UN_VOTES_DB_PATH), for services/un_votes.py.

Export formats differ between collections and over time, so this script
does not guess which column is which. Step 1 prints the columns; step 2
imports with every mapping given explicitly. The CSV is expected in
"long" form: one row per (resolution, country).

Step 1 - look at the file:
    venv\\Scripts\\python.exe scripts\\load_un_votes.py --csv votes.csv --show-columns

Step 2 - import, naming the columns and how vote values map to Y/N/A/'':
    venv\\Scripts\\python.exe scripts\\load_un_votes.py --csv votes.csv ^
        --col-symbol <col> --col-date <col> --col-title <col> ^
        --col-country <col> --col-vote <col> ^
        --vote-map "Y=Y,N=N,A=A,X=" ^
        [--col-adopted-without-vote <col> --adopted-true-values "<v1>,<v2>"]

Every vote value found in the file must appear in --vote-map; unknown
values stop the import (listing them) rather than being dropped. Without
--col-adopted-without-vote, adopted_without_vote is stored as NULL
(unknown). Re-importing the same resolution replaces its rows.
"""

from __future__ import annotations

import argparse
import csv
import sys
from contextlib import closing
from pathlib import Path
from typing import Dict, Iterable, List, Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config import get_settings  # noqa: E402
from services.un_votes import connect, create_schema, normalize_un_symbol  # noqa: E402

settings = get_settings()

ALLOWED_VOTES = {"Y", "N", "A", ""}


def read_rows(path: Path) -> List[dict]:
    # utf-8-sig drops the BOM Excel-made CSVs often start with.
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def show_columns(rows: List[dict], sample: int = 3) -> None:
    if not rows:
        print("The file has a header but no rows.")
        return
    print("Columns:")
    for name in rows[0]:
        examples = " | ".join(str(r.get(name, ""))[:40] for r in rows[:sample])
        print(f"  {name!r}: {examples}")
    print(f"\n{len(rows)} rows.")


def parse_vote_map(spec: str) -> Dict[str, str]:
    """"Y=Y,N=N,X=" -> {"Y": "Y", "N": "N", "X": ""}."""
    mapping = {}
    for pair in spec.split(","):
        source, sep, target = pair.partition("=")
        if not sep or target.strip() not in ALLOWED_VOTES:
            raise ValueError(f"bad --vote-map entry {pair!r}: use VALUE=Y|N|A|(empty)")
        mapping[source.strip()] = target.strip()
    return mapping


def unmapped_votes(rows: Iterable[dict], column: str, vote_map: Dict[str, str]) -> List[str]:
    return sorted({r[column].strip() for r in rows} - set(vote_map))


def missing_columns(rows: List[dict], columns: Iterable[Optional[str]]) -> List[str]:
    header = set(rows[0]) if rows else set()
    return [c for c in columns if c and c not in header]


def _symbol(raw: str) -> str:
    return normalize_un_symbol(raw) or raw.strip()


def _adopted(row: dict, args: argparse.Namespace) -> Optional[bool]:
    if not args.col_adopted_without_vote:
        return None
    true_values = {v.strip() for v in args.adopted_true_values.split(",")}
    return row[args.col_adopted_without_vote].strip() in true_values


def import_rows(conn, rows: List[dict], args: argparse.Namespace, vote_map: Dict[str, str]) -> tuple:
    """Insert resolutions and votes; returns (resolution count, vote count)."""
    create_schema(conn)
    resolutions = {}
    for row in rows:
        symbol = _symbol(row[args.col_symbol])
        resolutions[symbol] = (symbol, row[args.col_date].strip(), row[args.col_title].strip(), _adopted(row, args))
    votes = [
        (_symbol(row[args.col_symbol]), row[args.col_country].strip(), vote_map[row[args.col_vote].strip()])
        for row in rows
        if row[args.col_country].strip()
    ]
    with conn:
        conn.executemany("INSERT OR REPLACE INTO resolutions VALUES (?, ?, ?, ?)", resolutions.values())
        conn.executemany("INSERT OR REPLACE INTO votes VALUES (?, ?, ?)", votes)
    return len(resolutions), len(votes)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Import UN voting data (CSV) into SQLite.")
    parser.add_argument("--csv", required=True, type=Path)
    parser.add_argument("--db", type=Path, default=settings.un_votes_db_path)
    parser.add_argument("--show-columns", action="store_true", help="Print the columns and exit")
    for name in ("symbol", "date", "title", "country", "vote"):
        parser.add_argument(f"--col-{name}", help=f"CSV column holding the {name}")
    parser.add_argument("--vote-map", help='How vote values map, e.g. "Y=Y,N=N,A=A,X="')
    parser.add_argument("--col-adopted-without-vote", help="Optional column flagging adoption without a vote")
    parser.add_argument("--adopted-true-values", default="", help="Values of that column meaning 'yes'")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    rows = read_rows(args.csv)
    if args.show_columns:
        show_columns(rows)
        return 0

    required = [args.col_symbol, args.col_date, args.col_title, args.col_country, args.col_vote, args.vote_map]
    if not all(required):
        parser.error("give every --col-* option and --vote-map (run --show-columns first)")
    if args.col_adopted_without_vote and not args.adopted_true_values:
        parser.error("--col-adopted-without-vote needs --adopted-true-values")
    missing = missing_columns(rows, [*required[:5], args.col_adopted_without_vote])
    if missing:
        sys.exit(f"Column(s) not in the CSV: {missing}")
    vote_map = parse_vote_map(args.vote_map)
    unknown = unmapped_votes(rows, args.col_vote, vote_map)
    if unknown:
        sys.exit(f"Vote values missing from --vote-map: {unknown}")

    args.db.parent.mkdir(parents=True, exist_ok=True)
    with closing(connect(args.db)) as conn:
        n_res, n_votes = import_rows(conn, rows, args, vote_map)
    print(f"Imported {n_res} resolution(s) and {n_votes} vote(s) into {args.db}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
