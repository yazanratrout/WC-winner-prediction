"""
Player-Level Features

Computes per-team aggregates from the players table:
  - avg_squad_rating       — mean rating across all players (top 23 by rating)
  - gk_rating              — best GK rating
  - def_rating             — avg rating of top-4 defenders
  - mid_rating             — avg rating of top-5 midfielders
  - fwd_rating             — avg rating of top-3 forwards
  - squad_depth            — mean rating of players 12–23 (bench quality)
  - injury_impact          — sum of injured players' ratings (negative signal)
  - avg_age                — mean age across squad
  - market_value_total_m   — total squad market value in millions
  - caps_proxy             — avg caps (falls back to 0 if caps column is empty)

Run:
  python -m src.features.player_features
"""

import sqlite3

import numpy as np
import pandas as pd

from src.data.database import get_connection

_SQUAD_SIZE = 23

_POSITION_MAP = {
    "GK": ["GK", "goalkeeper", "portiere"],
    "DEF": ["CB", "LB", "RB", "LWB", "RWB", "DEF", "defender", "difensore", "D"],
    "MID": ["CM", "CAM", "CDM", "LM", "RM", "MID", "midfielder", "centrocampista", "M"],
    "FWD": ["ST", "CF", "LW", "RW", "FWD", "forward", "attaccante", "F", "SS"],
}


def _classify_position(pos_str: str | None) -> str:
    if not pos_str:
        return "MID"
    s = pos_str.upper().split(",")[0].strip()
    for label, variants in _POSITION_MAP.items():
        if any(s.startswith(v.upper()) for v in variants):
            return label
    return "MID"


def compute_player_features(conn: sqlite3.Connection) -> pd.DataFrame:
    """Return one row per team with player-aggregate features."""
    players = pd.read_sql_query(
        "SELECT team, position, rating, market_value_m, age, caps, injured "
        "FROM players",
        conn,
    )

    players["pos_group"] = players["position"].apply(_classify_position)

    output = []
    for team, grp in players.groupby("team"):
        grp = grp.copy()

        # Top-23 by rating for squad
        top23 = grp.nlargest(_SQUAD_SIZE, "rating", keep="all").head(_SQUAD_SIZE)
        avg_squad = float(top23["rating"].mean()) if not top23.empty else 50.0

        # Positional top picks
        def pos_avg(pos_label: str, n: int) -> float:
            sub = grp[grp["pos_group"] == pos_label].nlargest(n, "rating")
            return float(sub["rating"].mean()) if not sub.empty else avg_squad

        gk_rating  = pos_avg("GK", 1)
        def_rating = pos_avg("DEF", 4)
        mid_rating = pos_avg("MID", 5)
        fwd_rating = pos_avg("FWD", 3)

        # Bench depth: players ranked 12–23
        bench = top23.iloc[11:] if len(top23) > 11 else pd.DataFrame()
        squad_depth = float(bench["rating"].mean()) if not bench.empty else avg_squad * 0.85

        # Injury impact: sum of ratings of currently injured players
        injured = grp[grp["injured"] == 1]
        injury_impact = float(injured["rating"].sum()) if not injured.empty else 0.0

        avg_age = float(grp["age"].dropna().mean()) if grp["age"].notna().any() else 26.0

        mv = grp["market_value_m"].dropna()
        market_value_total_m = float(mv.sum()) if not mv.empty else 0.0

        caps = grp["caps"].dropna()
        caps_proxy = float(caps.mean()) if not caps.empty else 0.0

        output.append({
            "team":                  team,
            "avg_squad_rating":      avg_squad,
            "gk_rating":             gk_rating,
            "def_rating":            def_rating,
            "mid_rating":            mid_rating,
            "fwd_rating":            fwd_rating,
            "squad_depth":           squad_depth,
            "injury_impact":         injury_impact,
            "avg_age":               avg_age,
            "market_value_total_m":  market_value_total_m,
            "caps_proxy":            caps_proxy,
        })

    result_df = pd.DataFrame(output)
    print(f"  Player features computed for {len(result_df)} teams")
    return result_df


def run() -> pd.DataFrame:
    print("=== Player-Level Features ===")
    conn = get_connection()
    df = compute_player_features(conn)
    conn.close()
    print("\nSample (top 10 by avg squad rating):")
    print(
        df.nlargest(10, "avg_squad_rating")[
            ["team", "avg_squad_rating", "gk_rating", "def_rating",
             "mid_rating", "fwd_rating", "squad_depth", "avg_age",
             "market_value_total_m"]
        ].to_string(index=False)
    )
    print("\nPlayer features complete.")
    return df


if __name__ == "__main__":
    run()
