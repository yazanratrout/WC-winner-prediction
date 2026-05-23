"""
Load WC 2026 fixture data from data/raw/worldcup2026.db into the main DB.

Creates two tables:
  wc2026_groups   — team, group_letter, fifa_code
  wc2026_bracket  — match_number, stage, home_label, away_label, match_label

Run:
  python -m src.data.load_wc2026
"""

import sqlite3
from pathlib import Path

import pandas as pd

from src.data.database import get_connection
from src.data.team_names import canonical

SRC_DB = Path("data/raw/worldcup2026.db")


def _ensure_tables(conn: sqlite3.Connection) -> None:
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS wc2026_groups (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            team         TEXT NOT NULL,
            fifa_code    TEXT,
            group_letter TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS wc2026_bracket (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            match_number INTEGER NOT NULL,
            stage        TEXT NOT NULL,
            stage_order  INTEGER NOT NULL,
            home_label   TEXT,
            away_label   TEXT,
            match_label  TEXT,
            kickoff_at   TEXT,
            city         TEXT
        );
    """)
    conn.commit()


def load_wc2026(conn: sqlite3.Connection) -> None:
    src = sqlite3.connect(SRC_DB)
    src.row_factory = sqlite3.Row

    # --- Groups ---
    teams_df = pd.read_sql("SELECT team_name, fifa_code, group_letter FROM teams ORDER BY group_letter, id", src)
    teams_df["team"] = teams_df["team_name"].apply(canonical)

    conn.execute("DELETE FROM wc2026_groups")
    for _, row in teams_df.iterrows():
        conn.execute(
            "INSERT INTO wc2026_groups (team, fifa_code, group_letter) VALUES (?,?,?)",
            (row["team"], row["fifa_code"], row["group_letter"]),
        )

    print(f"  Loaded {len(teams_df)} teams into wc2026_groups")

    # --- Bracket ---
    matches_df = pd.read_sql(
        """
        SELECT m.match_number, ts.stage_name, ts.stage_order,
               t1.team_name AS home_name, t2.team_name AS away_name,
               m.match_label, m.kickoff_at, hc.city_name
        FROM matches m
        JOIN tournament_stages ts ON m.stage_id = ts.id
        LEFT JOIN teams t1 ON m.home_team_id = t1.id
        LEFT JOIN teams t2 ON m.away_team_id = t2.id
        LEFT JOIN host_cities hc ON m.city_id = hc.id
        ORDER BY ts.stage_order, m.match_number
        """,
        src,
    )

    conn.execute("DELETE FROM wc2026_bracket")
    for _, row in matches_df.iterrows():
        home_label = canonical(row["home_name"]) if pd.notna(row["home_name"]) else None
        away_label = canonical(row["away_name"]) if pd.notna(row["away_name"]) else None
        conn.execute(
            """INSERT INTO wc2026_bracket
               (match_number, stage, stage_order, home_label, away_label, match_label, kickoff_at, city)
               VALUES (?,?,?,?,?,?,?,?)""",
            (
                int(row["match_number"]),
                row["stage_name"],
                int(row["stage_order"]),
                home_label,
                away_label,
                row["match_label"],
                row["kickoff_at"],
                row["city_name"],
            ),
        )

    conn.commit()
    src.close()

    group_matches = len(matches_df[matches_df["stage_order"] == 1])
    knockout_matches = len(matches_df[matches_df["stage_order"] > 1])
    print(f"  Loaded {group_matches} group stage + {knockout_matches} knockout fixtures into wc2026_bracket")


def run() -> None:
    print("=== Loading WC 2026 fixture data ===")
    conn = get_connection()
    _ensure_tables(conn)
    load_wc2026(conn)
    conn.close()
    print("Done.")


if __name__ == "__main__":
    run()
