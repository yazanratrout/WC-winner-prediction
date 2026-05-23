"""
Tactical & xG Features

Estimates per-team tactical signals from match data alone (no tracking data):
  - xg_scored_10          — proxy xG per match over last 10 (goals + 0.3*shots_proxy)
  - xg_conceded_10        — same for goals conceded
  - shot_creation_rate    — goals_scored as shot proxy (goals / matches, last 10)
  - conversion_efficiency — goals / expected_attacks proxy
  - set_piece_efficiency  — close-match goals as set-piece proxy (can be enriched later)
  - pressing_proxy        — high-scoring wins ratio (teams that press win big)
  - possession_proxy      — derived from goal diff + clean sheets (indirect signal)

All values are per-team averages; exact values should be treated as ordinal
signals, not ground-truth statistics.

Run:
  python -m src.features.tactical_features
"""

import sqlite3

import numpy as np
import pandas as pd

from src.data.database import get_connection


def compute_tactical_features(conn: sqlite3.Connection) -> pd.DataFrame:
    matches = pd.read_sql_query(
        "SELECT date, home_team, away_team, home_score, away_score "
        "FROM matches WHERE home_score IS NOT NULL ORDER BY date",
        conn,
    )

    # Expand to per-team view
    rows = []
    for _, m in matches.iterrows():
        for side in ("home", "away"):
            gs = m["home_score"] if side == "home" else m["away_score"]
            gc = m["away_score"] if side == "home" else m["home_score"]
            rows.append({
                "date": m["date"],
                "team": m["home_team"] if side == "home" else m["away_team"],
                "gs":   gs,
                "gc":   gc,
            })

    df = pd.DataFrame(rows).sort_values(["team", "date"]).reset_index(drop=True)

    output = []
    for team, grp in df.groupby("team"):
        grp = grp.sort_values("date").reset_index(drop=True)
        gs_all = grp["gs"].tolist()
        gc_all = grp["gc"].tolist()
        n = len(gs_all)

        last10_gs = gs_all[-10:] if n >= 10 else gs_all
        last10_gc = gc_all[-10:] if n >= 10 else gc_all

        # xG proxy: goals + small bonus for high-scoring matches
        xg_scored   = float(np.mean([g + 0.3 * max(g - 1, 0) for g in last10_gs])) if last10_gs else 0.0
        xg_conceded = float(np.mean([g + 0.3 * max(g - 1, 0) for g in last10_gc])) if last10_gc else 0.0

        # Shot creation proxy: goals per match (teams that shoot more score more)
        shot_creation = float(np.mean(last10_gs)) if last10_gs else 0.0

        # Conversion efficiency: goals relative to xG proxy (>1 = clinical)
        conversion = (float(np.mean(last10_gs)) / xg_scored) if xg_scored > 0 else 1.0

        # Pressing proxy: fraction of last-10 wins with >= 3 goals scored
        pressing = float(np.mean([1 if g >= 3 else 0 for g in last10_gs])) if last10_gs else 0.0

        # Set-piece efficiency proxy: fraction of matches scoring from behind
        # (can't derive from goals alone — use average goals scored in draws/losses as proxy)
        comeback = float(np.mean([
            1 if gs >= gc else 0
            for gs, gc in zip(last10_gs, last10_gc)
        ])) if last10_gs else 0.5

        # Possession proxy: teams with high GD tend to control the ball
        gd_vals = [gs - gc for gs, gc in zip(gs_all[-20:], gc_all[-20:])]
        avg_gd  = float(np.mean(gd_vals)) if gd_vals else 0.0
        possession_proxy = 0.5 + np.clip(avg_gd / 10.0, -0.3, 0.3)

        output.append({
            "team":                 team,
            "xg_scored_10":         xg_scored,
            "xg_conceded_10":       xg_conceded,
            "shot_creation_rate":   shot_creation,
            "conversion_efficiency": conversion,
            "set_piece_efficiency": comeback,
            "pressing_proxy":       pressing,
            "possession_proxy":     possession_proxy,
        })

    result_df = pd.DataFrame(output)
    print(f"  Tactical features computed for {len(result_df)} teams")
    return result_df


def run() -> pd.DataFrame:
    print("=== Tactical & xG Features ===")
    conn = get_connection()
    df = compute_tactical_features(conn)
    conn.close()
    print("\nSample (top 10 by xg_scored_10):")
    print(
        df.nlargest(10, "xg_scored_10")[
            ["team", "xg_scored_10", "xg_conceded_10", "shot_creation_rate",
             "conversion_efficiency", "pressing_proxy", "possession_proxy"]
        ].to_string(index=False)
    )
    print("\nTactical features complete.")
    return df


if __name__ == "__main__":
    run()
