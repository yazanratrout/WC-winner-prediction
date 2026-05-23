"""
Load and clean Kaggle historical datasets into SQLite.

Expected raw files (place in data/raw/):
  results.csv         — international football results (Kaggle: martj42)
  fifa_ranking.csv    — FIFA ranking history (Kaggle: cashncarry)
  players.csv         — FIFA player data (Kaggle: stefanoleone992 or similar)

Run:
  python -m src.data.load_kaggle
"""

import sqlite3
from pathlib import Path

import pandas as pd

from src.data.database import get_connection, init_db
from src.data.team_names import canonical

RAW = Path("data/raw")
CUTOFF_YEAR = 2014  # only keep matches from this year forward for training


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _upsert_teams(conn: sqlite3.Connection, teams: set[str]) -> None:
    c = conn.cursor()
    for t in sorted(teams):
        c.execute(
            "INSERT OR IGNORE INTO teams (name) VALUES (?)",
            (t,),
        )
    conn.commit()


# ---------------------------------------------------------------------------
# Match results
# ---------------------------------------------------------------------------

def load_match_results(conn: sqlite3.Connection) -> int:
    path = RAW / "results.csv"
    if not path.exists():
        print(f"  SKIP  {path} not found — download from Kaggle first")
        return 0

    df = pd.read_csv(path, parse_dates=["date"])

    # Filter to recent years
    df = df[df["date"].dt.year >= CUTOFF_YEAR].copy()

    # Canonical names
    df["home_team"] = df["home_team"].map(canonical)
    df["away_team"] = df["away_team"].map(canonical)

    # Outcome from home perspective
    def outcome(row):
        if row["home_score"] > row["away_score"]:
            return "W"
        elif row["home_score"] == row["away_score"]:
            return "D"
        return "L"

    # Drop rows without scores (future/unplayed matches)
    df = df.dropna(subset=["home_score", "away_score"])
    df["home_score"] = df["home_score"].astype(int)
    df["away_score"] = df["away_score"].astype(int)

    df["outcome"] = df.apply(outcome, axis=1)
    df["neutral"] = df["neutral"].astype(int)
    df["source"] = "kaggle"

    # Ensure teams exist
    all_teams = set(df["home_team"]) | set(df["away_team"])
    _upsert_teams(conn, all_teams)

    # Insert matches (skip duplicates by date + teams)
    c = conn.cursor()
    inserted = 0
    for _, row in df.iterrows():
        try:
            c.execute(
                """
                INSERT INTO matches
                  (date, home_team, away_team, home_score, away_score,
                   competition, neutral, outcome, source)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(row["date"].date()),
                    row["home_team"],
                    row["away_team"],
                    int(row["home_score"]),
                    int(row["away_score"]),
                    row.get("tournament", "unknown"),
                    int(row["neutral"]),
                    row["outcome"],
                    "kaggle",
                ),
            )
            inserted += 1
        except sqlite3.IntegrityError:
            pass

    conn.commit()
    print(f"  Loaded {inserted:,} matches from results.csv")
    return inserted


# ---------------------------------------------------------------------------
# FIFA rankings
# ---------------------------------------------------------------------------

def load_fifa_rankings(conn: sqlite3.Connection) -> int:
    path = RAW / "fifa_ranking.csv"
    if not path.exists():
        print(f"  SKIP  {path} not found")
        return 0

    df = pd.read_csv(path, parse_dates=["rank_date"])
    df = df[df["rank_date"].dt.year >= CUTOFF_YEAR].copy()

    # Common column name variations
    rank_col = next((c for c in df.columns if "rank" in c.lower() and "date" not in c.lower()), None)
    points_col = next((c for c in df.columns if "point" in c.lower()), None)
    team_col = next((c for c in df.columns if "country" in c.lower() or "team" in c.lower()), None)

    if not all([rank_col, team_col]):
        print("  SKIP  fifa_ranking.csv column format not recognised")
        return 0

    df = df.dropna(subset=[rank_col])
    df["team"] = df[team_col].map(canonical)
    c = conn.cursor()
    inserted = 0
    for _, row in df.iterrows():
        c.execute(
            "INSERT INTO fifa_rankings (date, team, rank, points) VALUES (?, ?, ?, ?)",
            (
                str(row["rank_date"].date()),
                row["team"],
                int(row[rank_col]),
                float(row[points_col]) if points_col else None,
            ),
        )
        inserted += 1

    conn.commit()
    print(f"  Loaded {inserted:,} FIFA ranking rows")
    return inserted


# ---------------------------------------------------------------------------
# Players
# ---------------------------------------------------------------------------

def load_players(conn: sqlite3.Connection) -> int:
    path = RAW / "players.csv"
    if not path.exists():
        print(f"  SKIP  {path} not found")
        return 0

    df = pd.read_csv(path, low_memory=False)

    # Map source columns to canonical names — use first match only to avoid duplicate columns
    candidates = {
        "name":           ["short_name", "long_name", "player_name"],
        "team":           ["nationality_name", "nationality", "nation"],
        "club":           ["club_name", "club"],
        "rating":         ["overall", "player_overall_rating"],
        "market_value_m": ["value_eur"],
        "age":            ["age"],
        "position":       ["player_positions", "position"],
    }
    rename = {}
    for target, sources in candidates.items():
        for src in sources:
            if src in df.columns and target not in df.columns and src not in rename:
                rename[src] = target
                break
    df = df.rename(columns=rename)

    required = {"name", "team"}
    if not required.issubset(df.columns):
        print("  SKIP  players.csv missing required columns (name, team)")
        return 0

    df["team"] = df["team"].map(canonical)
    if "market_value_m" in df.columns:
        df["market_value_m"] = pd.to_numeric(df["market_value_m"], errors="coerce") / 1_000_000

    season = str(df["fifa_version"].iloc[0]) if "fifa_version" in df.columns else "unknown"

    all_teams = set(df["team"].dropna())
    _upsert_teams(conn, all_teams)

    c = conn.cursor()
    inserted = 0
    for _, row in df.iterrows():
        c.execute(
            """
            INSERT INTO players
              (name, team, position, rating, market_value_m, age, club, season)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                row["name"] if "name" in row.index else None,
                row["team"] if "team" in row.index else None,
                row["position"] if "position" in row.index else None,
                float(row["rating"]) if "rating" in row.index and pd.notna(row["rating"]) else None,
                float(row["market_value_m"]) if "market_value_m" in row.index and pd.notna(row["market_value_m"]) else None,
                int(row["age"]) if "age" in row.index and pd.notna(row["age"]) else None,
                row["club"] if "club" in row.index else None,
                str(season),
            ),
        )
        inserted += 1

    conn.commit()
    print(f"  Loaded {inserted:,} player rows")
    return inserted


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run() -> None:
    print("=== Loading Kaggle historical data ===")
    init_db()
    conn = get_connection()
    load_match_results(conn)
    load_fifa_rankings(conn)
    load_players(conn)
    conn.close()
    print("Done.")


if __name__ == "__main__":
    run()
