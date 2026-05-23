"""
Data cleaning and unification.

Merges match results with FIFA rankings at the time of each match,
validates data quality, and produces clean unified views in the DB.

Run:
  python -m src.data.clean_unify
"""

import sqlite3
from datetime import datetime

import pandas as pd

from src.data.database import get_connection
from src.data.team_names import canonical


# ---------------------------------------------------------------------------
# Check and fix team name consistency
# ---------------------------------------------------------------------------

def fix_team_name_consistency(conn: sqlite3.Connection) -> None:
    """Apply canonical name mapping to any remaining non-standard names in DB."""
    c = conn.cursor()

    c.execute("SELECT DISTINCT home_team FROM matches")
    home_teams = [r[0] for r in c.fetchall()]
    c.execute("SELECT DISTINCT away_team FROM matches")
    away_teams = [r[0] for r in c.fetchall()]

    all_names = set(home_teams + away_teams)
    fixed = 0
    for name in all_names:
        canon = canonical(name)
        if canon != name:
            c.execute("UPDATE matches SET home_team = ? WHERE home_team = ?", (canon, name))
            c.execute("UPDATE matches SET away_team = ? WHERE away_team = ?", (canon, name))
            c.execute("UPDATE players SET team = ? WHERE team = ?", (canon, name))
            c.execute("UPDATE teams SET name = ? WHERE name = ?", (canon, name))
            fixed += 1

    conn.commit()
    if fixed:
        print(f"  Fixed {fixed} non-canonical team names")
    else:
        print("  All team names are canonical")


# ---------------------------------------------------------------------------
# Merge rankings into matches (snapshot at match date)
# ---------------------------------------------------------------------------

def attach_rankings_to_matches(conn: sqlite3.Connection) -> None:
    """
    For each match, find the most recent FIFA ranking entry before the match
    date for both teams and store them in a materialized view table.
    """
    matches_df = pd.read_sql("SELECT id, date, home_team, away_team FROM matches", conn)
    rankings_df = pd.read_sql("SELECT date, team, rank FROM fifa_rankings", conn)

    if rankings_df.empty:
        print("  SKIP  No FIFA ranking data available yet")
        return

    rankings_df["date"] = pd.to_datetime(rankings_df["date"])
    matches_df["date"] = pd.to_datetime(matches_df["date"])

    def latest_rank(team: str, before_date: pd.Timestamp) -> int | None:
        sub = rankings_df[(rankings_df["team"] == team) & (rankings_df["date"] <= before_date)]
        if sub.empty:
            return None
        return int(sub.sort_values("date").iloc[-1]["rank"])

    c = conn.cursor()
    # Create the merged view table if it doesn't exist
    c.execute("""
        CREATE TABLE IF NOT EXISTS match_rankings (
            match_id        INTEGER PRIMARY KEY,
            home_rank       INTEGER,
            away_rank       INTEGER,
            rank_diff       INTEGER
        )
    """)

    updated = 0
    for _, row in matches_df.iterrows():
        home_rank = latest_rank(row["home_team"], row["date"])
        away_rank = latest_rank(row["away_team"], row["date"])
        rank_diff = (home_rank - away_rank) if (home_rank and away_rank) else None

        c.execute("""
            INSERT OR REPLACE INTO match_rankings (match_id, home_rank, away_rank, rank_diff)
            VALUES (?, ?, ?, ?)
        """, (row["id"], home_rank, away_rank, rank_diff))
        updated += 1

    conn.commit()
    print(f"  Attached rankings to {updated:,} matches")


# ---------------------------------------------------------------------------
# Validate data quality
# ---------------------------------------------------------------------------

def validate_data(conn: sqlite3.Connection) -> dict:
    results = {}

    c = conn.cursor()

    # Total counts
    c.execute("SELECT COUNT(*) FROM matches")
    results["total_matches"] = c.fetchone()[0]

    c.execute("SELECT COUNT(*) FROM teams")
    results["total_teams"] = c.fetchone()[0]

    c.execute("SELECT COUNT(*) FROM players")
    results["total_players"] = c.fetchone()[0]

    c.execute("SELECT COUNT(*) FROM fifa_rankings")
    results["total_ranking_rows"] = c.fetchone()[0]

    # Null checks
    c.execute("SELECT COUNT(*) FROM matches WHERE home_score IS NULL OR away_score IS NULL")
    results["matches_with_null_score"] = c.fetchone()[0]

    c.execute("SELECT COUNT(*) FROM players WHERE rating IS NULL")
    results["players_with_null_rating"] = c.fetchone()[0]

    # Date range
    c.execute("SELECT MIN(date), MAX(date) FROM matches")
    row = c.fetchone()
    results["match_date_range"] = f"{row[0]} → {row[1]}"

    # Top teams by match count
    c.execute("""
        SELECT t, COUNT(*) as n FROM (
            SELECT home_team as t FROM matches
            UNION ALL
            SELECT away_team as t FROM matches
        ) GROUP BY t ORDER BY n DESC LIMIT 5
    """)
    results["top_5_teams_by_matches"] = [(r[0], r[1]) for r in c.fetchall()]

    print("\n=== Data Quality Report ===")
    for k, v in results.items():
        print(f"  {k}: {v}")
    print()

    return results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run() -> None:
    print("=== Data cleaning and unification ===")
    conn = get_connection()
    fix_team_name_consistency(conn)
    attach_rankings_to_matches(conn)
    validate_data(conn)
    conn.close()
    print("Done.")


if __name__ == "__main__":
    run()
