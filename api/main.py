"""
FastAPI backend

Endpoints:
  GET  /teams                      — all 48 WC 2026 teams with current win%
  GET  /team/{name}                — single team profile + probabilities
  GET  /group/{letter}             — group standings + team probabilities
  POST /match/predict              — predict W/D/L for any two teams
  POST /simulate                   — trigger a fresh tournament simulation
  POST /match/result               — submit a real match result + update
  GET  /history/{team}             — probability evolution over time
  GET  /health                     — liveness check

Run:
  source venv/bin/activate
  uvicorn api.main:app --reload --port 8000
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.data.database import get_connection
from src.simulation.predictor import MatchPredictor

# ── Shared state loaded once at startup ─────────────────────────────────────
_predictor: MatchPredictor | None = None
_scheduler = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _predictor, _scheduler
    print("[api] Loading match predictor...")
    _predictor = MatchPredictor()

    print("[api] Starting live sync scheduler...")
    try:
        from src.data.updater import start_scheduler
        _scheduler = start_scheduler()
    except Exception as e:
        print(f"[api] Scheduler failed to start (non-fatal): {e}")

    yield  # app runs

    if _scheduler:
        _scheduler.shutdown(wait=False)


app = FastAPI(
    title="WC 2026 Prediction Engine",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten in production
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _conn():
    return get_connection()


def _team_or_404(name: str, conn) -> dict:
    row = conn.execute(
        "SELECT name, elo, fifa_rank, confederation FROM teams WHERE name=?", (name,)
    ).fetchone()
    if not row:
        raise HTTPException(404, f"Team '{name}' not found")
    return dict(row)


def _sim_row(team: str, conn) -> dict | None:
    row = conn.execute(
        "SELECT qualify_pct, r16_pct, qf_pct, sf_pct, final_pct, winner_pct "
        "FROM simulations WHERE team=?", (team,)
    ).fetchone()
    return dict(row) if row else None


def _load_team_feats(team: str, conn) -> dict:
    """Reconstruct a team's raw feature vector from the features parquet."""
    import numpy as np
    import pandas as pd
    from collections import defaultdict

    df = pd.read_parquet("data/processed/features.parquet")
    diff_cols  = [c for c in df.columns if c.startswith("diff_") and c != "diff_best_rank"]
    feat_names = [c[5:] for c in diff_cols]

    team_vals: dict[str, list] = defaultdict(list)
    for _, row in df.iterrows():
        ht, at = row["home_team"], row["away_team"]
        for dc, fn in zip(diff_cols, feat_names):
            v = row[dc]
            if pd.notna(v):
                if ht == team:
                    team_vals[fn].append(float(v))
                elif at == team:
                    team_vals[fn].append(-float(v))

    result = {fn: float(np.mean(vals)) if vals else 0.0
              for fn, vals in team_vals.items()}

    rank_row = conn.execute(
        "SELECT f.rank FROM fifa_rankings f "
        "INNER JOIN (SELECT team, MAX(date) AS md FROM fifa_rankings WHERE team=?) t "
        "ON f.team=t.team AND f.date=t.md",
        (team,)
    ).fetchone()
    result["best_rank"] = rank_row["rank"] if rank_row else 200
    return result


# ── Pydantic models ───────────────────────────────────────────────────────────

class MatchPredictRequest(BaseModel):
    home_team: str
    away_team: str
    neutral: bool = True


class MatchResultRequest(BaseModel):
    home_team: str
    away_team: str
    home_score: int
    away_score: int
    competition: str = "FIFA World Cup"
    neutral: bool = True
    match_date: str | None = None


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "timestamp": datetime.now(timezone.utc).isoformat()}


@app.get("/teams")
def get_teams():
    """All 48 WC 2026 teams sorted by winner probability."""
    conn = _conn()
    rows = conn.execute(
        """
        SELECT g.team, g.group_letter, g.fifa_code,
               t.elo,
               s.qualify_pct, s.r16_pct, s.qf_pct,
               s.sf_pct, s.final_pct, s.winner_pct
        FROM wc2026_groups g
        LEFT JOIN teams t ON t.name = g.team
        LEFT JOIN simulations s ON s.team = g.team
        ORDER BY s.winner_pct DESC NULLS LAST
        """
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


@app.get("/team/{name}")
def get_team(name: str):
    """Single team profile with probabilities and recent Elo history."""
    conn = _conn()
    team = _team_or_404(name, conn)
    sim  = _sim_row(name, conn)

    # Group info
    grp = conn.execute(
        "SELECT group_letter, fifa_code FROM wc2026_groups WHERE team=?", (name,)
    ).fetchone()

    # Last 10 Elo history points
    elo_history = conn.execute(
        "SELECT date, elo FROM elo_history WHERE team=? ORDER BY date DESC LIMIT 10",
        (name,)
    ).fetchall()

    # Group opponents
    opponents = []
    if grp:
        opp_rows = conn.execute(
            "SELECT team FROM wc2026_groups WHERE group_letter=? AND team!=?",
            (grp["group_letter"], name),
        ).fetchall()
        opponents = [r["team"] for r in opp_rows]

    conn.close()
    return {
        "team":         name,
        "fifa_code":    grp["fifa_code"] if grp else None,
        "group":        grp["group_letter"] if grp else None,
        "elo":          round(team["elo"] or 1500, 1),
        "opponents":    opponents,
        "probabilities": sim,
        "elo_history":  [dict(r) for r in elo_history],
    }


@app.get("/group/{letter}")
def get_group(letter: str):
    """All teams in a group with their probabilities."""
    letter = letter.upper()
    conn = _conn()
    rows = conn.execute(
        """
        SELECT g.team, g.fifa_code,
               t.elo,
               s.qualify_pct, s.r16_pct, s.qf_pct,
               s.sf_pct, s.final_pct, s.winner_pct
        FROM wc2026_groups g
        LEFT JOIN teams t ON t.name = g.team
        LEFT JOIN simulations s ON s.team = g.team
        WHERE g.group_letter = ?
        ORDER BY s.winner_pct DESC NULLS LAST
        """,
        (letter,),
    ).fetchall()
    conn.close()
    if not rows:
        raise HTTPException(404, f"Group '{letter}' not found")
    return {"group": letter, "teams": [dict(r) for r in rows]}


@app.post("/match/predict")
def predict_match(req: MatchPredictRequest):
    """Predict W/D/L probabilities for any two teams."""
    if _predictor is None:
        raise HTTPException(503, "Model not loaded yet")

    conn = _conn()
    _team_or_404(req.home_team, conn)
    _team_or_404(req.away_team, conn)

    fa = _load_team_feats(req.home_team, conn)
    fb = _load_team_feats(req.away_team, conn)

    elo_h = conn.execute(
        "SELECT elo FROM teams WHERE name=?", (req.home_team,)
    ).fetchone()["elo"] or 1500.0
    elo_a = conn.execute(
        "SELECT elo FROM teams WHERE name=?", (req.away_team,)
    ).fetchone()["elo"] or 1500.0
    conn.close()

    probs = _predictor.predict(fa, fb, elo_h, elo_a, neutral=req.neutral)

    return {
        "home_team":  req.home_team,
        "away_team":  req.away_team,
        "neutral":    req.neutral,
        "elo_diff":   round(elo_h - elo_a, 1),
        "probabilities": {
            "home_win": round(probs["W"] * 100, 1),
            "draw":     round(probs["D"] * 100, 1),
            "away_win": round(probs["L"] * 100, 1),
        },
    }


@app.post("/simulate")
def trigger_simulation(background_tasks: BackgroundTasks):
    """Kick off a fresh 10,000-run tournament simulation in the background."""
    def _run():
        from src.simulation.tournament import simulate_tournament, save_results
        from src.data.updater import snapshot_probabilities
        conn = get_connection()
        df = simulate_tournament(conn, N=10_000)
        save_results(df, conn, N=10_000)
        snapshot_probabilities(conn)
        conn.close()

    background_tasks.add_task(_run)
    return {"status": "simulation started", "n": 10_000}


@app.post("/match/result")
def submit_result(req: MatchResultRequest, background_tasks: BackgroundTasks):
    """
    Submit a real match result. Updates Elo and re-simulates in the background.
    Returns immediately; use GET /teams to see updated probabilities.
    """
    from src.data.updater import update_after_match

    def _run():
        update_after_match(
            home_team=req.home_team,
            away_team=req.away_team,
            home_score=req.home_score,
            away_score=req.away_score,
            competition=req.competition,
            neutral=req.neutral,
            match_date=req.match_date,
        )

    background_tasks.add_task(_run)
    return {
        "status":    "update queued",
        "match":     f"{req.home_team} {req.home_score}–{req.away_score} {req.away_team}",
        "note":      "probabilities will update in ~15s",
    }


@app.get("/history/{team}")
def get_history(team: str, stage: str = "winner"):
    """
    Probability evolution for a team over time.
    stage: qualify | r16 | qf | sf | final | winner
    """
    valid_stages = {"qualify", "r16", "qf", "sf", "final", "winner"}
    if stage not in valid_stages:
        raise HTTPException(400, f"stage must be one of {valid_stages}")

    conn = _conn()
    _team_or_404(team, conn)

    rows = conn.execute(
        """
        SELECT timestamp, probability
        FROM probability_history
        WHERE team=? AND stage=?
        ORDER BY timestamp ASC
        """,
        (team, stage),
    ).fetchall()
    conn.close()

    return {
        "team":    team,
        "stage":   stage,
        "history": [{"timestamp": r["timestamp"],
                     "probability": r["probability"]} for r in rows],
    }
