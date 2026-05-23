"""
Team-Level Features

Computes and stores:
  - Elo ratings (updated per match, written to elo_history + teams tables)
  - Recent form score (last 5 + last 10, exponentially weighted)
  - Average goals scored / conceded (last 10 matches)
  - Home vs away performance splits
  - Performance vs top-20 FIFA ranked opponents
  - Clean sheet rate and goal difference trend (last 10)
  - Continental tournament performance score

All functions return a dict keyed by team name so downstream code can
build feature rows without re-querying.

Run:
  python -m src.features.team_features
"""

import sqlite3
from collections import defaultdict
from datetime import datetime

import numpy as np
import pandas as pd

from src.data.database import get_connection

# ---------------------------------------------------------------------------
# Elo constants
# ---------------------------------------------------------------------------
ELO_K = 32          # standard K-factor
ELO_START = 1500.0  # default starting Elo
HOME_ADVANTAGE = 100  # points added to home team expected score


def _elo_expected(rating_a: float, rating_b: float) -> float:
    return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400.0))


def _elo_update(rating: float, expected: float, actual: float) -> float:
    return rating + ELO_K * (actual - expected)


# ---------------------------------------------------------------------------
# Elo ratings
# ---------------------------------------------------------------------------

def compute_elo(conn: sqlite3.Connection) -> dict[str, float]:
    """Walk every match chronologically and update Elo for both teams.

    Returns final Elo dict {team: elo}.
    Writes each match's before-Elo to matches.home_elo_before / away_elo_before.
    Writes every state transition to elo_history.
    Updates teams.elo with final values.
    """
    df = pd.read_sql_query(
        "SELECT id, date, home_team, away_team, home_score, away_score, neutral "
        "FROM matches WHERE home_score IS NOT NULL ORDER BY date, id",
        conn,
    )

    elo: dict[str, float] = defaultdict(lambda: ELO_START)
    history_rows = []

    c = conn.cursor()
    c.execute("DELETE FROM elo_history")

    for _, row in df.iterrows():
        ht, at = row["home_team"], row["away_team"]
        r_h, r_a = elo[ht], elo[at]

        # Apply home advantage for non-neutral venues
        adj_h = r_h + (0 if row["neutral"] else HOME_ADVANTAGE)
        exp_h = _elo_expected(adj_h, r_a)
        exp_a = 1.0 - exp_h

        hs, as_ = row["home_score"], row["away_score"]
        if hs > as_:
            act_h, act_a = 1.0, 0.0
        elif hs == as_:
            act_h, act_a = 0.5, 0.5
        else:
            act_h, act_a = 0.0, 1.0

        # Store before-match Elo in matches table
        c.execute(
            "UPDATE matches SET home_elo_before=?, away_elo_before=? WHERE id=?",
            (r_h, r_a, int(row["id"])),
        )

        new_h = _elo_update(r_h, exp_h, act_h)
        new_a = _elo_update(r_a, exp_a, act_a)
        elo[ht], elo[at] = new_h, new_a

        history_rows.append((row["date"], ht, new_h))
        history_rows.append((row["date"], at, new_a))

    c.executemany("INSERT INTO elo_history (date, team, elo) VALUES (?,?,?)", history_rows)

    for team, rating in elo.items():
        c.execute("UPDATE teams SET elo=? WHERE name=?", (rating, team))

    conn.commit()
    print(f"  Elo computed for {len(elo)} teams, {len(history_rows)//2} match transitions written")
    return dict(elo)


# ---------------------------------------------------------------------------
# Form / goals / home-away / vs-top20 / clean-sheets
# ---------------------------------------------------------------------------

def compute_team_form_stats(conn: sqlite3.Connection) -> pd.DataFrame:
    """Compute rolling stats for every team using all available match history.

    Returns one row per team with columns:
      form_5, form_10          — exponentially weighted W=1 D=0.5 L=0 over last 5/10
      goals_scored_10          — avg goals scored in last 10
      goals_conceded_10        — avg goals conceded in last 10
      gd_trend_10              — avg goal difference in last 10
      clean_sheet_rate_10      — fraction of last 10 with 0 goals conceded
      home_win_rate            — win rate in home matches (all history)
      away_win_rate            — win rate in away matches (all history)
      vs_top20_win_rate        — win rate vs top-20 FIFA ranked opponents
    """
    matches = pd.read_sql_query(
        "SELECT date, home_team, away_team, home_score, away_score, neutral "
        "FROM matches WHERE home_score IS NOT NULL ORDER BY date",
        conn,
    )
    rankings = pd.read_sql_query(
        "SELECT date, team, rank FROM fifa_rankings ORDER BY date",
        conn,
    )

    # Build a helper: for each match date, what was each team's rank?
    # We'll use a simple forward-fill: merge on nearest rank_date <= match_date
    rankings["date"] = pd.to_datetime(rankings["date"])
    matches["date"] = pd.to_datetime(matches["date"])

    # Latest rank snapshot before or on a given date
    def get_rank_on(team: str, date) -> int | None:
        sub = rankings[(rankings["team"] == team) & (rankings["date"] <= date)]
        if sub.empty:
            return None
        return int(sub.iloc[-1]["rank"])

    # Expand matches into per-team perspective rows
    rows = []
    for _, m in matches.iterrows():
        for side in ("home", "away"):
            team = m["home_team"] if side == "home" else m["away_team"]
            opp  = m["away_team"] if side == "home" else m["home_team"]
            gs   = m["home_score"] if side == "home" else m["away_score"]
            gc   = m["away_score"] if side == "home" else m["home_score"]
            if gs > gc:
                res = "W"
            elif gs == gc:
                res = "D"
            else:
                res = "L"
            rows.append({
                "date":    m["date"],
                "team":    team,
                "opp":     opp,
                "gs":      gs,
                "gc":      gc,
                "result":  res,
                "is_home": side == "home",
                "neutral": m["neutral"],
            })

    df = pd.DataFrame(rows).sort_values(["team", "date"]).reset_index(drop=True)

    # Exponential weights for last N
    def exp_form(results: list[str], n: int) -> float:
        recent = results[-n:]
        weights = np.exp(np.linspace(0, 1, len(recent)))
        weights /= weights.sum()
        scores = [1.0 if r == "W" else 0.5 if r == "D" else 0.0 for r in recent]
        return float(np.dot(weights, scores))

    # Pre-build rank lookup cache for unique (team, date) pairs to avoid
    # calling get_rank_on in the tight loop below
    all_teams = df["opp"].unique()
    all_dates  = matches["date"].unique()
    # Build dict: (team, date_str) -> rank
    rank_cache: dict[tuple, int | None] = {}
    latest_rank: dict[str, int | None] = {}
    for _, r in rankings.sort_values("date").iterrows():
        latest_rank[r["team"]] = int(r["rank"])

    # For opponent rank at match time: group rankings by team and use searchsorted
    rank_by_team: dict[str, pd.DataFrame] = {}
    for team, grp in rankings.groupby("team"):
        rank_by_team[team] = grp.sort_values("date").reset_index(drop=True)

    def opp_rank_on(opp: str, date) -> int | None:
        if opp not in rank_by_team:
            return None
        grp = rank_by_team[opp]
        idx = grp["date"].searchsorted(date, side="right") - 1
        if idx < 0:
            return None
        return int(grp.iloc[idx]["rank"])

    output = []
    for team, grp in df.groupby("team"):
        grp = grp.sort_values("date").reset_index(drop=True)
        results_list = grp["result"].tolist()
        gs_list      = grp["gs"].tolist()
        gc_list      = grp["gc"].tolist()

        form_5  = exp_form(results_list, 5)  if len(results_list) >= 1 else 0.5
        form_10 = exp_form(results_list, 10) if len(results_list) >= 1 else 0.5

        last10_gs = gs_list[-10:] if len(gs_list) >= 10 else gs_list
        last10_gc = gc_list[-10:] if len(gc_list) >= 10 else gc_list
        last10_res = results_list[-10:] if len(results_list) >= 10 else results_list

        goals_scored_10   = float(np.mean(last10_gs)) if last10_gs else 0.0
        goals_conceded_10 = float(np.mean(last10_gc)) if last10_gc else 0.0
        gd_trend_10       = goals_scored_10 - goals_conceded_10
        clean_sheet_rate  = float(np.mean([1 if g == 0 else 0 for g in last10_gc])) if last10_gc else 0.0

        home_rows = grp[grp["is_home"] & (grp["neutral"] == 0)]
        away_rows = grp[~grp["is_home"] & (grp["neutral"] == 0)]
        home_win_rate = float((home_rows["result"] == "W").mean()) if len(home_rows) else 0.5
        away_win_rate = float((away_rows["result"] == "W").mean()) if len(away_rows) else 0.5

        # Performance vs top-20
        top20_rows = []
        for _, r in grp.iterrows():
            opp_rank = opp_rank_on(r["opp"], r["date"])
            if opp_rank is not None and opp_rank <= 20:
                top20_rows.append(r["result"])
        vs_top20_win_rate = float(np.mean([1 if r == "W" else 0 for r in top20_rows])) if top20_rows else 0.0

        output.append({
            "team":                team,
            "form_5":              form_5,
            "form_10":             form_10,
            "goals_scored_10":     goals_scored_10,
            "goals_conceded_10":   goals_conceded_10,
            "gd_trend_10":         gd_trend_10,
            "clean_sheet_rate_10": clean_sheet_rate,
            "home_win_rate":       home_win_rate,
            "away_win_rate":       away_win_rate,
            "vs_top20_win_rate":   vs_top20_win_rate,
        })

    result_df = pd.DataFrame(output)
    print(f"  Form/stats computed for {len(result_df)} teams")
    return result_df


# ---------------------------------------------------------------------------
# Continental tournament performance
# ---------------------------------------------------------------------------

_CONTINENTAL_KEYWORDS = [
    "copa america", "afcon", "africa cup", "euro", "asian cup", "gold cup",
    "concacaf", "afc", "ofc nations", "nations cup",
]


def compute_continental_score(conn: sqlite3.Connection) -> pd.DataFrame:
    """Win rate in continental tournament matches (all history)."""
    matches = pd.read_sql_query(
        "SELECT home_team, away_team, outcome, competition "
        "FROM matches WHERE home_score IS NOT NULL",
        conn,
    )
    matches["competition_lc"] = matches["competition"].str.lower().fillna("")
    is_continental = matches["competition_lc"].apply(
        lambda c: any(kw in c for kw in _CONTINENTAL_KEYWORDS)
    )
    cont = matches[is_continental].copy()

    rows = []
    for _, m in cont.iterrows():
        rows.append({"team": m["home_team"], "result": m["outcome"]})
        # away perspective
        away_res = "W" if m["outcome"] == "L" else ("L" if m["outcome"] == "W" else "D")
        rows.append({"team": m["away_team"], "result": away_res})

    if not rows:
        teams = pd.read_sql_query("SELECT name AS team FROM teams", conn)
        teams["continental_score"] = 0.0
        return teams

    df = pd.DataFrame(rows)
    scores = (
        df.groupby("team")["result"]
        .apply(lambda rs: float(np.mean([1 if r == "W" else 0.5 if r == "D" else 0 for r in rs])))
        .reset_index()
        .rename(columns={"result": "continental_score"})
    )
    print(f"  Continental score computed from {len(cont)} continental matches")
    return scores


# ---------------------------------------------------------------------------
# Entry point — runs all 2.1 computations and prints a sample
# ---------------------------------------------------------------------------

def run() -> None:
    print("=== Team-Level Features ===")
    conn = get_connection()

    elo_dict = compute_elo(conn)
    form_df  = compute_team_form_stats(conn)
    cont_df  = compute_continental_score(conn)

    # Merge into one team features frame for inspection
    team_features = form_df.merge(cont_df, on="team", how="left")
    team_features["elo"] = team_features["team"].map(elo_dict)

    print("\nSample (top 10 by Elo):")
    print(
        team_features.sort_values("elo", ascending=False)
        .head(10)[["team", "elo", "form_10", "goals_scored_10", "goals_conceded_10",
                    "gd_trend_10", "clean_sheet_rate_10", "vs_top20_win_rate",
                    "continental_score"]]
        .to_string(index=False)
    )

    conn.close()
    print("\nTeam features complete.")
    return team_features


if __name__ == "__main__":
    run()
