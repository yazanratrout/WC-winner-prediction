"""
Squad Chemistry Features

Computes intra-squad familiarity signals:
  - same_club_pairs        — count of player pairs sharing a club in the squad
  - avg_shared_club_mins   — proxy: normalised same_club_pairs / squad_size
  - lineup_stability       — not derivable from player table alone; set to 0
                             (will be populated if starting XI data is available)
  - shared_apps_proxy      — mean caps across squad (proxy for shared intl experience)
  - def_familiarity        — same-club count restricted to defenders + GK
  - chemistry_score        — weighted composite of the above (0–1 scale)

Run:
  python -m src.features.chemistry_features
"""

import sqlite3

import numpy as np
import pandas as pd

from src.data.database import get_connection
from src.features.player_features import _classify_position

_SQUAD_SIZE = 23

_DEF_GROUPS = {"GK", "DEF"}


def compute_chemistry(conn: sqlite3.Connection) -> pd.DataFrame:
    players = pd.read_sql_query(
        "SELECT team, position, club, caps FROM players",
        conn,
    )
    players["pos_group"] = players["position"].apply(_classify_position)

    output = []
    for team, grp in players.groupby("team"):
        top23 = grp.nlargest(_SQUAD_SIZE, "caps" if "caps" in grp.columns else grp.columns[0],
                             keep="all").head(_SQUAD_SIZE)
        n = len(top23)
        clubs = top23["club"].dropna().tolist()

        # Same-club pair count
        club_counts: dict[str, int] = {}
        for c in clubs:
            club_counts[c] = club_counts.get(c, 0) + 1
        same_club_pairs = int(sum(v * (v - 1) // 2 for v in club_counts.values()))

        # Normalised club cohesion
        max_pairs = n * (n - 1) // 2 if n > 1 else 1
        avg_shared_club = same_club_pairs / max_pairs

        # Defensive familiarity
        def_grp = top23[top23["pos_group"].isin(_DEF_GROUPS)]
        def_clubs = def_grp["club"].dropna().tolist()
        def_club_counts: dict[str, int] = {}
        for c in def_clubs:
            def_club_counts[c] = def_club_counts.get(c, 0) + 1
        def_same = int(sum(v * (v - 1) // 2 for v in def_club_counts.values()))
        def_n = len(def_clubs)
        def_max = def_n * (def_n - 1) // 2 if def_n > 1 else 1
        def_familiarity = def_same / def_max

        # Shared intl experience proxy
        caps_vals = top23["caps"].dropna()
        shared_apps_proxy = float(caps_vals.mean()) if not caps_vals.empty else 0.0
        # Normalise caps to 0–1 (cap at 100 appearances)
        shared_apps_norm = min(shared_apps_proxy / 100.0, 1.0)

        # Composite chemistry score (weights sum to 1)
        chemistry_score = (
            0.35 * avg_shared_club
            + 0.35 * def_familiarity
            + 0.30 * shared_apps_norm
        )

        output.append({
            "team":               team,
            "same_club_pairs":    same_club_pairs,
            "avg_shared_club":    avg_shared_club,
            "lineup_stability":   0.0,  # placeholder — needs live lineup history
            "shared_apps_proxy":  shared_apps_proxy,
            "def_familiarity":    def_familiarity,
            "chemistry_score":    chemistry_score,
        })

    result_df = pd.DataFrame(output)
    print(f"  Chemistry features computed for {len(result_df)} teams")
    return result_df


def run() -> pd.DataFrame:
    print("=== Squad Chemistry Features ===")
    conn = get_connection()
    df = compute_chemistry(conn)
    conn.close()
    print("\nSample (top 10 by chemistry_score):")
    print(
        df.nlargest(10, "chemistry_score")[
            ["team", "same_club_pairs", "avg_shared_club", "def_familiarity",
             "shared_apps_proxy", "chemistry_score"]
        ].to_string(index=False)
    )
    print("\nChemistry features complete.")
    return df


if __name__ == "__main__":
    run()
