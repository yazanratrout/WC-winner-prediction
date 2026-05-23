"""
Differential Feature Pipeline

Builds the final training dataset:
  1. Loads all historical matches with known outcomes
  2. For each match, fetches the team-level features AS OF that date
     (rolling features are pre-computed up to that match only — no leakage)
  3. Computes team_A minus team_B differentials for every feature
  4. Adds meta columns: neutral flag, competition type
  5. Exports to data/processed/features.parquet

The differential approach means the model learns "how much better is team A
than team B on each dimension", not raw team values.

Run:
  python -m src.features.build_features
"""

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.database import get_connection
from src.features.team_features import (
    compute_elo,
    compute_team_form_stats,
    compute_continental_score,
    ELO_START,
)
from src.features.player_features import compute_player_features
from src.features.chemistry_features import compute_chemistry
from src.features.tactical_features import compute_tactical_features

OUT_PATH = Path("data/processed/features.parquet")

_COMP_CATEGORIES = {
    "world_cup":      ["FIFA World Cup", "World Cup"],
    "continental":    ["UEFA Euro", "Copa America", "AFCON", "Asian Cup", "Gold Cup", "Nations Cup"],
    "qualifier":      ["qualifier", "Qualification", "qualifying"],
    "friendly":       ["friendly", "Friendlies", "International Friendlies"],
    "nations_league": ["UEFA Nations League", "Nations League"],
}


def _comp_type(competition: str) -> str:
    if not competition:
        return "other"
    for cat, keywords in _COMP_CATEGORIES.items():
        if any(kw.lower() in competition.lower() for kw in keywords):
            return cat
    return "other"


def _build_rolling_elo(conn: sqlite3.Connection) -> dict[str, dict[str, float]]:
    """Returns {match_id: {home_elo, away_elo}} from pre-computed elo_before columns."""
    df = pd.read_sql_query(
        "SELECT id, home_team, away_team, home_elo_before, away_elo_before FROM matches",
        conn,
    )
    result = {}
    for _, r in df.iterrows():
        result[int(r["id"])] = {
            "home_elo": r["home_elo_before"] if pd.notna(r["home_elo_before"]) else ELO_START,
            "away_elo": r["away_elo_before"] if pd.notna(r["away_elo_before"]) else ELO_START,
        }
    return result


def build_features(conn: sqlite3.Connection) -> pd.DataFrame:
    """Build differential feature matrix for all labelled matches."""

    # --- Team-level feature tables (global, computed on all history) ---
    # These are "final state" features — ok for the overall team profile.
    # For training we also attach rolling Elo per-match (no leakage).
    print("  Computing team form stats...")
    form_df    = compute_team_form_stats(conn)
    print("  Computing player features...")
    player_df  = compute_player_features(conn)
    print("  Computing chemistry features...")
    chem_df    = compute_chemistry(conn)
    print("  Computing tactical features...")
    tact_df    = compute_tactical_features(conn)
    print("  Computing continental scores...")
    cont_df    = compute_continental_score(conn)

    # Merge all into one team profile table
    team_df = form_df
    for extra in [player_df, chem_df, tact_df, cont_df]:
        team_df = team_df.merge(extra, on="team", how="left")

    # Elo per team (final)
    elo_series = pd.read_sql_query("SELECT name AS team, elo FROM teams", conn)
    team_df = team_df.merge(elo_series, on="team", how="left")
    team_df["elo"] = team_df["elo"].fillna(ELO_START)

    # FIFA rank per team (most recent)
    rank_df = pd.read_sql_query(
        "SELECT f.team, f.rank AS best_rank "
        "FROM fifa_rankings f "
        "INNER JOIN (SELECT team, MAX(date) AS max_date FROM fifa_rankings GROUP BY team) t "
        "ON f.team = t.team AND f.date = t.max_date",
        conn,
    )
    team_df = team_df.merge(rank_df, on="team", how="left")
    team_df["best_rank"] = team_df["best_rank"].fillna(200)  # unranked → 200

    team_idx = team_df.set_index("team")

    # Per-match rolling Elo (from pre-computed home_elo_before / away_elo_before)
    rolling_elo = _build_rolling_elo(conn)

    # Load all labelled matches
    matches = pd.read_sql_query(
        "SELECT id, date, home_team, away_team, home_score, away_score, "
        "       neutral, competition, outcome "
        "FROM matches WHERE outcome IS NOT NULL ORDER BY date",
        conn,
    )

    feature_cols = [
        c for c in team_df.columns if c != "team"
    ]

    rows = []
    for _, m in matches.iterrows():
        ht, at = m["home_team"], m["away_team"]
        if ht not in team_idx.index or at not in team_idx.index:
            continue

        h = team_idx.loc[ht]
        a = team_idx.loc[at]

        # Differential row
        diff = {}
        for col in feature_cols:
            try:
                diff[f"diff_{col}"] = float(h[col]) - float(a[col])
            except (TypeError, ValueError):
                diff[f"diff_{col}"] = 0.0

        # Override Elo diff with per-match rolling Elo
        mid = int(m["id"])
        if mid in rolling_elo:
            diff["diff_elo"] = rolling_elo[mid]["home_elo"] - rolling_elo[mid]["away_elo"]

        diff["neutral"]       = int(m["neutral"])
        diff["comp_type"]     = _comp_type(m["competition"])
        diff["date"]          = m["date"]
        diff["home_team"]     = ht
        diff["away_team"]     = at
        diff["outcome"]       = m["outcome"]  # W/D/L from home perspective
        diff["home_score"]    = m["home_score"]
        diff["away_score"]    = m["away_score"]
        rows.append(diff)

    df = pd.DataFrame(rows)

    # Encode competition type as integers
    comp_map = {"world_cup": 5, "continental": 4, "nations_league": 3,
                "qualifier": 2, "friendly": 1, "other": 0}
    df["comp_type_enc"] = df["comp_type"].map(comp_map).fillna(0).astype(int)
    df = df.drop(columns=["comp_type"])

    # Closeness features — absolute value of key differentials.
    # High value = evenly matched → draw more likely.
    # These let the model learn "when are teams too close to call?"
    for col in ["diff_elo", "diff_form_10", "diff_gd_trend_10",
                "diff_goals_scored_10", "diff_goals_conceded_10"]:
        if col in df.columns:
            df[f"abs_{col}"] = df[col].abs()

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_PATH, index=False)
    print(f"\n  Exported {len(df):,} rows × {len(df.columns)} columns → {OUT_PATH}")
    return df


def run() -> pd.DataFrame:
    print("=== Differential Feature Pipeline ===")
    conn = get_connection()

    # Ensure Elo is populated before building features
    print("  Running Elo computation first...")
    compute_elo(conn)

    df = build_features(conn)
    conn.close()

    print("\nFeature matrix sample (first 5 rows, key columns):")
    key_cols = ["date", "home_team", "away_team", "outcome", "diff_elo",
                "diff_form_10", "diff_avg_squad_rating", "diff_gd_trend_10",
                "neutral", "comp_type_enc"]
    available = [c for c in key_cols if c in df.columns]
    print(df[available].head().to_string(index=False))

    print("\nOutcome distribution:")
    print(df["outcome"].value_counts())

    print("\nBuild complete — features.parquet ready.")
    return df


if __name__ == "__main__":
    run()
