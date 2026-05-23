"""
Dynamic update pipeline

update_after_match()
  Called whenever a real WC match result comes in.
  Steps:
    1. Insert the match result into the matches table
    2. Re-run Elo update for the two teams
    3. Re-simulate the tournament (10,000 runs)
    4. Snapshot the new probability table into probability_history

snapshot_probabilities()
  Reads current simulations table and writes a timestamped copy
  into probability_history — used by the /history/{team} endpoint.

Concurrency: a threading.Lock prevents two updates running simultaneously.
"""

from __future__ import annotations

import sqlite3
import threading
from datetime import datetime, timezone

from src.data.database import get_connection
from src.features.team_features import (
    ELO_K, ELO_START, _elo_expected, _elo_update,
)

_update_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Elo update for a single match result
# ---------------------------------------------------------------------------

def _apply_elo_result(
    conn: sqlite3.Connection,
    home_team: str,
    away_team: str,
    home_score: int,
    away_score: int,
    neutral: bool = True,
) -> tuple[float, float]:
    """Update Elo for both teams and return (new_home_elo, new_away_elo)."""
    c = conn.cursor()
    c.execute("SELECT elo FROM teams WHERE name=?", (home_team,))
    row = c.fetchone()
    elo_h = row["elo"] if row else ELO_START

    c.execute("SELECT elo FROM teams WHERE name=?", (away_team,))
    row = c.fetchone()
    elo_a = row["elo"] if row else ELO_START

    if home_score > away_score:
        act_h, act_a = 1.0, 0.0
    elif home_score == away_score:
        act_h, act_a = 0.5, 0.5
    else:
        act_h, act_a = 0.0, 1.0

    exp_h = _elo_expected(elo_h, elo_a)
    new_h = _elo_update(elo_h, exp_h, act_h)
    new_a = _elo_update(elo_a, 1 - exp_h, act_a)

    c.execute("UPDATE teams SET elo=?, updated_at=? WHERE name=?",
              (new_h, datetime.now(timezone.utc).isoformat(), home_team))
    c.execute("UPDATE teams SET elo=?, updated_at=? WHERE name=?",
              (new_a, datetime.now(timezone.utc).isoformat(), away_team))

    date_str = datetime.now(timezone.utc).date().isoformat()
    c.execute("INSERT INTO elo_history (date, team, elo) VALUES (?,?,?)",
              (date_str, home_team, new_h))
    c.execute("INSERT INTO elo_history (date, team, elo) VALUES (?,?,?)",
              (date_str, away_team, new_a))
    conn.commit()
    return new_h, new_a


# ---------------------------------------------------------------------------
# Probability snapshot
# ---------------------------------------------------------------------------

def snapshot_probabilities(conn: sqlite3.Connection) -> int:
    """Copy current simulations table into probability_history."""
    c = conn.cursor()
    c.execute("SELECT team, qualify_pct, r16_pct, qf_pct, sf_pct, final_pct, winner_pct FROM simulations")
    rows = c.fetchall()
    if not rows:
        return 0

    ts = datetime.now(timezone.utc).isoformat()
    inserted = 0
    for row in rows:
        for stage, pct in [
            ("qualify", row["qualify_pct"]),
            ("r16",     row["r16_pct"]),
            ("qf",      row["qf_pct"]),
            ("sf",      row["sf_pct"]),
            ("final",   row["final_pct"]),
            ("winner",  row["winner_pct"]),
        ]:
            c.execute(
                "INSERT INTO probability_history (timestamp, team, stage, probability) VALUES (?,?,?,?)",
                (ts, row["team"], stage, pct),
            )
            inserted += 1
    conn.commit()
    return inserted


# ---------------------------------------------------------------------------
# update_after_match()
# ---------------------------------------------------------------------------

def update_after_match(
    home_team: str,
    away_team: str,
    home_score: int,
    away_score: int,
    competition: str = "FIFA World Cup",
    neutral: bool = True,
    match_date: str | None = None,
) -> dict:
    """
    Process a new match result end-to-end:
      1. Insert match into DB (skip if duplicate)
      2. Update Elo for both teams
      3. Re-run full tournament simulation
      4. Snapshot new probabilities to history
      5. Return summary dict

    Thread-safe — only one update runs at a time.
    """
    if not _update_lock.acquire(timeout=30):
        return {"error": "update already in progress, try again shortly"}

    try:
        conn = get_connection()
        date_str = match_date or datetime.now(timezone.utc).date().isoformat()

        # 1. Insert match
        if home_score > away_score:
            outcome = "W"
        elif home_score == away_score:
            outcome = "D"
        else:
            outcome = "L"

        c = conn.cursor()
        try:
            c.execute(
                """INSERT INTO matches
                   (date, home_team, away_team, home_score, away_score,
                    competition, neutral, outcome, source)
                   VALUES (?,?,?,?,?,?,?,?,?)""",
                (date_str, home_team, away_team, home_score, away_score,
                 competition, int(neutral), outcome, "live"),
            )
            conn.commit()
            inserted = True
        except sqlite3.IntegrityError:
            inserted = False

        # 2. Elo update
        new_h, new_a = _apply_elo_result(
            conn, home_team, away_team, home_score, away_score, neutral
        )

        # 3. Re-simulate
        from src.simulation.tournament import simulate_tournament, save_results
        df = simulate_tournament(conn, N=10_000)
        save_results(df, conn, N=10_000)

        # 4. Snapshot
        n_snapped = snapshot_probabilities(conn)

        conn.close()

        # Pull top-5 winners from results
        top5 = df.head(5)[["team", "winner_pct"]].to_dict("records")

        return {
            "status":        "ok",
            "match_inserted": inserted,
            "home_team":     home_team,
            "away_team":     away_team,
            "score":         f"{home_score}–{away_score}",
            "new_elo":       {home_team: round(new_h, 1), away_team: round(new_a, 1)},
            "top5_winners":  top5,
            "history_rows":  n_snapped,
        }

    finally:
        _update_lock.release()


# ---------------------------------------------------------------------------
# APScheduler live sync job
# ---------------------------------------------------------------------------

def start_scheduler() -> None:
    """
    Start a background scheduler that polls the live API every hour
    and triggers update_after_match() for any newly finished WC matches.
    """
    from apscheduler.schedulers.background import BackgroundScheduler
    from src.data.api_client import fetch_recent_fixtures

    scheduler = BackgroundScheduler()

    def _sync_job():
        try:
            conn = get_connection()
            # Fetch WC 2026 league (ID 1 = World Cup)
            fetch_recent_fixtures(conn, league_id=1, season=2026)
            conn.close()
            print(f"[scheduler] live sync complete {datetime.now(timezone.utc).isoformat()}")
        except Exception as e:
            print(f"[scheduler] sync error: {e}")

    scheduler.add_job(_sync_job, "interval", hours=1, id="live_sync")
    scheduler.start()
    print("[scheduler] started — polling every hour")
    return scheduler
