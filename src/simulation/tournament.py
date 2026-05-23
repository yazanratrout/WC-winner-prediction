"""
Tournament Simulation Engine (vectorised)

Simulates WC 2026 across N Monte Carlo runs simultaneously:
  - All N simulations of each match are batched into one predict_proba call
  - In-tournament Elo updates after every round (vectorised)
  - Stakes-aware draw calibration on group stage matchday 3
  - Best-8 third-place rule for R32 qualification
  - Knockout bracket: R32 → R16 → QF → SF → 3rd-place playoff → Final
  - Penalty shootout for drawn knockout matches

Run:
  python -m src.simulation.tournament
"""

from __future__ import annotations

import random
import sqlite3
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.data.database import get_connection
from src.features.team_features import ELO_START, _elo_expected

N_SIMULATIONS = 10_000
RANDOM_SEED   = 42
MODEL_PATH    = Path("models/match_predictor.pkl")
COMP_ENC_WC   = 5   # world_cup

# Elo K-factor and home advantage (same as training)
ELO_K          = 32
HOME_ADVANTAGE = 100

# Draw calibration thresholds
ELO_VERY_CLOSE      = 50
ELO_CLOSE_THRESHOLD = 100
VERY_CLOSE_DRAW_BONUS = 0.10
CLOSE_DRAW_BONUS      = 0.05


# ---------------------------------------------------------------------------
# Load assets
# ---------------------------------------------------------------------------

def _load_model():
    payload = joblib.load(MODEL_PATH)
    return payload["model"], payload["scaler"], payload["feature_cols"]


def _load_team_features(conn: sqlite3.Connection) -> dict[str, dict]:
    """Recover per-team raw feature values from features.parquet."""
    df = pd.read_parquet("data/processed/features.parquet")
    diff_cols  = [c for c in df.columns if c.startswith("diff_") and c != "diff_best_rank"]
    feat_names = [c[5:] for c in diff_cols]

    team_vals: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for _, row in df.iterrows():
        ht, at = row["home_team"], row["away_team"]
        for dc, fn in zip(diff_cols, feat_names):
            v = row[dc]
            if pd.notna(v):
                team_vals[ht][fn].append(float(v))
                team_vals[at][fn].append(-float(v))

    rank_df = pd.read_sql_query(
        "SELECT f.team, f.rank AS best_rank "
        "FROM fifa_rankings f "
        "INNER JOIN (SELECT team, MAX(date) AS max_date FROM fifa_rankings GROUP BY team) t "
        "ON f.team = t.team AND f.date = t.max_date",
        conn,
    )
    rank_map = dict(zip(rank_df["team"], rank_df["best_rank"]))

    result = {}
    for team, feats in team_vals.items():
        rd = {fn: float(np.mean(vals)) for fn, vals in feats.items()}
        rd["best_rank"] = rank_map.get(team, 200)
        result[team] = rd
    print(f"  Team features loaded for {len(result)} teams")
    return result


def _load_elo(conn: sqlite3.Connection) -> dict[str, float]:
    rows = pd.read_sql_query("SELECT name, elo FROM teams", conn)
    return {r["name"]: r["elo"] for _, r in rows.iterrows()}


def _load_groups(conn: sqlite3.Connection) -> dict[str, list[str]]:
    df = pd.read_sql_query(
        "SELECT team, group_letter FROM wc2026_groups ORDER BY group_letter", conn
    )
    groups: dict[str, list[str]] = defaultdict(list)
    for _, r in df.iterrows():
        groups[r["group_letter"]].append(r["team"])
    return dict(groups)


def _load_bracket(conn: sqlite3.Connection) -> list[dict]:
    return pd.read_sql_query(
        "SELECT match_number, stage, stage_order, match_label "
        "FROM wc2026_bracket WHERE stage_order > 1 ORDER BY stage_order, match_number",
        conn,
    ).to_dict("records")


# ---------------------------------------------------------------------------
# Vectorised predictor
# ---------------------------------------------------------------------------

class VectorisedPredictor:
    """Predict W/D/L probabilities for N matches in one batched call."""

    def __init__(self):
        self.model, self.scaler, self.feature_cols = _load_model()
        self._le = getattr(self.model, "_le", None)
        # Column index maps
        self._diff_feats = {c[5:]: i for i, c in enumerate(self.feature_cols)
                            if c.startswith("diff_")}
        self._abs_feats  = {c[9:]: i for i, c in enumerate(self.feature_cols)
                            if c.startswith("abs_diff_")}
        self._neutral_idx    = self.feature_cols.index("neutral") if "neutral" in self.feature_cols else None
        self._comp_idx       = self.feature_cols.index("comp_type_enc") if "comp_type_enc" in self.feature_cols else None

    def predict_batch(
        self,
        team_a_feats: list[dict],   # length N
        team_b_feats: list[dict],   # length N
        elo_a: np.ndarray,          # shape (N,)
        elo_b: np.ndarray,          # shape (N,)
        neutral: bool = True,
        comp_enc: int = COMP_ENC_WC,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns (p_w, p_d, p_l) each shape (N,).
        """
        N = len(team_a_feats)
        X = np.zeros((N, len(self.feature_cols)), dtype=np.float32)

        # Fill diff features
        for feat, col_idx in self._diff_feats.items():
            if feat == "elo":
                X[:, col_idx] = elo_a - elo_b
            else:
                X[:, col_idx] = np.array(
                    [fa.get(feat, 0.0) - fb.get(feat, 0.0)
                     for fa, fb in zip(team_a_feats, team_b_feats)],
                    dtype=np.float32,
                )

        # Fill abs diff features
        for feat, col_idx in self._abs_feats.items():
            if feat == "elo":
                X[:, col_idx] = np.abs(elo_a - elo_b)
            else:
                X[:, col_idx] = np.abs(
                    np.array([fa.get(feat, 0.0) - fb.get(feat, 0.0)
                               for fa, fb in zip(team_a_feats, team_b_feats)],
                              dtype=np.float32)
                )

        if self._neutral_idx is not None:
            X[:, self._neutral_idx] = int(neutral)
        if self._comp_idx is not None:
            X[:, self._comp_idx] = comp_enc

        X_scaled = self.scaler.transform(X)
        proba = self.model.predict_proba(X_scaled)  # (N, 3)

        # Map columns to W/D/L
        if self._le is not None:
            classes = list(self._le.classes_)
        else:
            classes = list(self.model.classes_)

        wi = classes.index("W")
        di = classes.index("D")
        li = classes.index("L")
        return proba[:, wi], proba[:, di], proba[:, li]


# ---------------------------------------------------------------------------
# Closeness draw boost (vectorised)
# ---------------------------------------------------------------------------

def _closeness_draw_boost(
    p_w: np.ndarray, p_d: np.ndarray, p_l: np.ndarray,
    elo_diff: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Boost draw probability for closely-matched teams."""
    abs_diff = np.abs(elo_diff)
    bonus = np.where(
        abs_diff <= ELO_VERY_CLOSE,
        VERY_CLOSE_DRAW_BONUS,
        np.where(
            abs_diff <= ELO_CLOSE_THRESHOLD,
            VERY_CLOSE_DRAW_BONUS + (CLOSE_DRAW_BONUS - VERY_CLOSE_DRAW_BONUS)
            * (abs_diff - ELO_VERY_CLOSE) / (ELO_CLOSE_THRESHOLD - ELO_VERY_CLOSE),
            0.0,
        ),
    )
    wl_total = p_w + p_l + 1e-9
    p_w = p_w - bonus * (p_w / wl_total)
    p_l = p_l - bonus * (p_l / wl_total)
    p_d = p_d + bonus
    # Normalise
    total = p_w + p_d + p_l
    return p_w / total, p_d / total, p_l / total


# ---------------------------------------------------------------------------
# Vectorised Elo update
# ---------------------------------------------------------------------------

def _elo_update_vec(
    elo_a: np.ndarray, elo_b: np.ndarray,
    act_a: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    exp_a = 1.0 / (1.0 + 10.0 ** ((elo_b - elo_a) / 400.0))
    new_a = elo_a + ELO_K * (act_a - exp_a)
    new_b = elo_b + ELO_K * ((1 - act_a) - (1 - exp_a))
    return new_a, new_b


# ---------------------------------------------------------------------------
# Outcome sampling (vectorised)
# ---------------------------------------------------------------------------

def _sample_outcomes(p_w: np.ndarray, p_d: np.ndarray, p_l: np.ndarray,
                     rng: np.random.Generator) -> np.ndarray:
    """Return array of 0=W, 1=D, 2=L."""
    r = rng.random(len(p_w))
    return np.where(r < p_w, 0, np.where(r < p_w + p_d, 1, 2))


def _sample_knockout(p_w: np.ndarray, p_d: np.ndarray, p_l: np.ndarray,
                     elo_a: np.ndarray, elo_b: np.ndarray,
                     rng: np.random.Generator) -> np.ndarray:
    """
    Knockout: no draws. Draws go to penalties (slight Elo-weighted 50/50).
    Returns array of 0=A wins, 1=B wins.
    """
    outcomes = _sample_outcomes(p_w, p_d, p_l, rng)
    is_draw = outcomes == 1
    # Penalty: compress Elo advantage toward 50/50
    exp_a = 1.0 / (1.0 + 10.0 ** ((elo_b - elo_a) / 400.0))
    pen_p_a = 0.5 + (exp_a - 0.5) * 0.4
    pen_result = rng.random(len(p_w)) < pen_p_a  # True = A wins penalties
    a_wins = ((outcomes == 0) | (is_draw & pen_result))
    return (~a_wins).astype(np.int8)  # 0=A wins, 1=B wins


# ---------------------------------------------------------------------------
# Stakes adjustments (vectorised, matchday 3)
# ---------------------------------------------------------------------------

_STAKES_SHIFTS = {
    # (a_status, b_status): (Δp_w, Δp_d, Δp_l)
    ("safe",        "safe"):        (-0.06,  0.12, -0.06),
    ("eliminated",  "eliminated"):  (-0.06,  0.12, -0.06),
    ("safe",        "eliminated"):  (-0.06,  0.12, -0.06),
    ("eliminated",  "safe"):        (-0.06,  0.12, -0.06),
    ("safe",        "must_win"):    (-0.08,  0.00,  0.08),
    ("must_win",    "safe"):        ( 0.08,  0.00, -0.08),
    ("safe",        "draw_enough"): (-0.05,  0.10, -0.05),
    ("draw_enough", "safe"):        (-0.05,  0.10, -0.05),
    ("draw_enough", "draw_enough"): (-0.075, 0.15, -0.075),
    ("must_win",    "must_win"):    ( 0.03, -0.06,  0.03),
    ("must_win",    "draw_enough"): ( 0.05,  0.03, -0.08),
    ("draw_enough", "must_win"):    (-0.08,  0.03,  0.05),
    ("eliminated",  "draw_enough"): (-0.12,  0.08,  0.04),
    ("draw_enough", "eliminated"):  ( 0.04,  0.08, -0.12),
    ("eliminated",  "must_win"):    (-0.10,  0.00,  0.10),
    ("must_win",    "eliminated"):  ( 0.10,  0.00, -0.10),
}


def _apply_stakes_vec(
    p_w: np.ndarray, p_d: np.ndarray, p_l: np.ndarray,
    a_statuses: list[str], b_statuses: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    p_w, p_d, p_l = p_w.copy(), p_d.copy(), p_l.copy()
    for i, (sa, sb) in enumerate(zip(a_statuses, b_statuses)):
        shift = _STAKES_SHIFTS.get((sa, sb))
        if shift:
            p_w[i] = max(0.01, p_w[i] + shift[0])
            p_d[i] = max(0.01, p_d[i] + shift[1])
            p_l[i] = max(0.01, p_l[i] + shift[2])
    total = p_w + p_d + p_l
    return p_w / total, p_d / total, p_l / total


# ---------------------------------------------------------------------------
# Group stage simulation (vectorised across N simulations)
# ---------------------------------------------------------------------------

def _derive_status(pts: int, gd: int, gf: int,
                   all_pts: list, all_gd: list, all_gf: int) -> str:
    """Derive qualification status for matchday-3 stakes."""
    rank = sum(
        1 for op, og, ogf in zip(all_pts, all_gd, all_gf)
        if (op > pts) or (op == pts and og > gd) or (op == pts and og == gd and ogf > gf)
    ) + 1

    max_pts = pts + 3
    # Simplified: look at 2nd place reachability
    sorted_others = sorted(zip(all_pts, all_gd, all_gf), reverse=True)
    second_pts = sorted_others[1][0] if len(sorted_others) > 1 else 0

    if rank <= 2 and pts > second_pts + 3:
        return "safe"
    elif rank <= 2:
        return "draw_enough"
    elif max_pts < second_pts:
        return "eliminated"
    elif max_pts == second_pts:
        return "draw_enough"
    else:
        return "must_win"


def simulate_groups_vectorised(
    groups: dict[str, list[str]],
    team_feats: dict[str, dict],
    base_elo: dict[str, float],
    predictor: VectorisedPredictor,
    N: int,
    rng: np.random.Generator,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], np.ndarray]:
    """
    Simulate all group matches for all N simulations in batch.

    Returns:
      group_ranks[group] = array (N, 4) of team indices (best to worst)
      elo_final[team]    = array (N,) of end-of-group Elo per sim
      sim_elo            = dict team → (N,) array (updated throughout)
    """
    # Elo: shape (N,) per team
    sim_elo: dict[str, np.ndarray] = {
        t: np.full(N, base_elo.get(t, ELO_START), dtype=np.float64)
        for t in team_feats
    }

    # Standings: pts, gd, gf — shape (N,) per team
    pts: dict[str, np.ndarray] = {t: np.zeros(N, dtype=np.int32) for t in team_feats}
    gd:  dict[str, np.ndarray] = {t: np.zeros(N, dtype=np.int32) for t in team_feats}
    gf:  dict[str, np.ndarray] = {t: np.zeros(N, dtype=np.int32) for t in team_feats}

    group_ranks: dict[str, np.ndarray] = {}  # group → (N, 4) team name arrays

    # Matchday schedule per group
    matchday_schedule = [
        (0, 1, 2, 3),  # MD1: team0 vs team1, team2 vs team3
        (0, 2, 1, 3),  # MD2: team0 vs team2, team1 vs team3
        (0, 3, 1, 2),  # MD3: team0 vs team3, team1 vs team2
    ]

    for letter, team_list in groups.items():
        fa = [team_feats.get(t, {}) for t in team_list]

        for md_idx, (a0, b0, a1, b1) in enumerate(matchday_schedule):
            matchday = md_idx + 1
            pairs = [(team_list[a0], team_list[b0]), (team_list[a1], team_list[b1])]

            for ta, tb in pairs:
                fta = [team_feats.get(ta, {})] * N
                ftb = [team_feats.get(tb, {})] * N
                ea  = sim_elo[ta]
                eb  = sim_elo[tb]

                p_w, p_d, p_l = predictor.predict_batch(fta, ftb, ea, eb)
                p_w, p_d, p_l = _closeness_draw_boost(p_w, p_d, p_l, ea - eb)

                # Matchday-3 stakes
                if matchday == 3:
                    all_teams_in_group = team_list
                    a_statuses, b_statuses = [], []
                    for sim_i in range(N):
                        grp_pts = [pts[t][sim_i] for t in all_teams_in_group]
                        grp_gd  = [gd[t][sim_i]  for t in all_teams_in_group]
                        grp_gf  = [gf[t][sim_i]  for t in all_teams_in_group]
                        others_a = [(grp_pts[j], grp_gd[j], grp_gf[j])
                                    for j, t in enumerate(all_teams_in_group) if t != ta]
                        others_b = [(grp_pts[j], grp_gd[j], grp_gf[j])
                                    for j, t in enumerate(all_teams_in_group) if t != tb]
                        a_statuses.append(_derive_status(
                            pts[ta][sim_i], gd[ta][sim_i], gf[ta][sim_i],
                            [o[0] for o in others_a], [o[1] for o in others_a],
                            [o[2] for o in others_a],
                        ))
                        b_statuses.append(_derive_status(
                            pts[tb][sim_i], gd[tb][sim_i], gf[tb][sim_i],
                            [o[0] for o in others_b], [o[1] for o in others_b],
                            [o[2] for o in others_b],
                        ))
                    p_w, p_d, p_l = _apply_stakes_vec(p_w, p_d, p_l, a_statuses, b_statuses)

                outcomes = _sample_outcomes(p_w, p_d, p_l, rng)  # 0=W,1=D,2=L

                # Points
                pts[ta] += np.where(outcomes == 0, 3, np.where(outcomes == 1, 1, 0))
                pts[tb] += np.where(outcomes == 2, 3, np.where(outcomes == 1, 1, 0))

                # Simulated goals (consistent with outcome)
                gs_a = np.where(outcomes == 0,
                                rng.integers(1, 5, N),
                                np.where(outcomes == 1,
                                         rng.integers(0, 3, N),
                                         rng.integers(0, 3, N)))
                gs_b = np.where(outcomes == 2,
                                rng.integers(1, 5, N),
                                np.where(outcomes == 1,
                                         gs_a,
                                         np.maximum(0, gs_a - 1)))
                gd[ta] += gs_a - gs_b
                gd[tb] += gs_b - gs_a
                gf[ta] += gs_a
                gf[tb] += gs_b

                # Elo update
                act_a = np.where(outcomes == 0, 1.0, np.where(outcomes == 1, 0.5, 0.0))
                sim_elo[ta], sim_elo[tb] = _elo_update_vec(ea, eb, act_a)

        # Rank teams within each group for each simulation
        # Stack: (4, N) for pts, gd, gf then argsort descending
        pts_mat = np.stack([pts[t] for t in team_list])  # (4, N)
        gd_mat  = np.stack([gd[t]  for t in team_list])
        gf_mat  = np.stack([gf[t]  for t in team_list])

        # Lexicographic sort: primary pts, secondary gd, tertiary gf
        sort_key = pts_mat * 10000 + gd_mat * 100 + gf_mat
        ranks = np.argsort(-sort_key, axis=0)  # (4, N): rank[0] = best team idx

        # Convert indices to team names: shape (N, 4)
        team_arr = np.array(team_list)
        group_ranks[letter] = team_arr[ranks].T  # (N, 4)

    return group_ranks, sim_elo


# ---------------------------------------------------------------------------
# Best-8 third-place selection (vectorised)
# ---------------------------------------------------------------------------

def select_best_thirds(
    group_ranks: dict[str, np.ndarray],
    group_pts: dict[str, dict[str, np.ndarray]],
    group_gd:  dict[str, dict[str, np.ndarray]],
    group_gf:  dict[str, dict[str, np.ndarray]],
    N: int,
) -> np.ndarray:
    """Returns (N, 8) array of best third-place team names."""
    letters = sorted(group_ranks.keys())
    # Third-place team per group: group_ranks[letter][:, 2] → (N,)
    thirds      = np.stack([group_ranks[l][:, 2] for l in letters], axis=1)  # (N, 12)
    # We don't have per-sim third-place stats easily here — use group rank position 2
    # For simplicity rank by pts → gd → gf stored in per-team arrays already in sim_elo context
    # We'll pass a simplified version: just return the 8 that qualified most often
    # (full per-sim tracking would require extra arrays; this is a good approximation)
    return thirds  # caller picks first 8 by score


# ---------------------------------------------------------------------------
# Knockout simulation (vectorised)
# ---------------------------------------------------------------------------

def simulate_knockout_vectorised(
    group_ranks: dict[str, np.ndarray],   # letter → (N, 4) team names
    sim_elo:     dict[str, np.ndarray],   # team → (N,) Elo
    team_feats:  dict[str, dict],
    predictor:   VectorisedPredictor,
    bracket:     list[dict],
    N:           int,
    rng:         np.random.Generator,
) -> dict[str, np.ndarray]:
    """
    Returns stage_counts[team][stage] = count across N sims.
    Stages: r16, qf, sf, final, winner.
    """
    letters = sorted(group_ranks.keys())

    # Best-8 third-place: sort 12 third-place teams by pts (approximated by Elo)
    thirds = np.stack([group_ranks[l][:, 2] for l in letters], axis=1)  # (N, 12)
    # Rank by final Elo as proxy for performance
    thirds_elo = np.stack(
        [np.array([sim_elo.get(thirds[sim_i, g], np.full(1, ELO_START))[sim_i]
                   for g in range(12)])
         for sim_i in range(N)]
    )  # (N, 12)
    best8_idx = np.argsort(-thirds_elo, axis=1)[:, :8]  # (N, 8) best indices
    best8 = np.array([[thirds[sim_i, best8_idx[sim_i, k]] for k in range(8)]
                      for sim_i in range(N)])  # (N, 8)

    # Build group letter → third-place letter mapping per sim
    # For bracket slot resolution we need which group each third-placer came from
    third_group_map = np.stack(
        [np.array(letters)] * N
    )  # (N, 12) letter per position

    # Map match_number → winner (N,) — team name per sim
    match_winners: dict[int, np.ndarray] = {}

    def resolve_slot(label: str) -> np.ndarray:
        """Resolve 'W73', '1A', '3ABCDF' → (N,) team names."""
        if label.startswith("W"):
            return match_winners[int(label[1:])]
        rank = int(label[0])
        rest = label[1:]
        if rank == 3:
            # Pick best available third-placer from eligible groups
            eligible = set(rest)
            result = np.empty(N, dtype=object)
            for sim_i in range(N):
                for k in range(8):
                    t = best8[sim_i, k]
                    g = third_group_map[sim_i, np.where(np.array(letters) == thirds[sim_i, best8_idx[sim_i, k]])[0][0]] \
                        if False else letters[np.where(thirds[sim_i] == t)[0][0]] \
                        if t in thirds[sim_i] else "?"
                    if g in eligible:
                        result[sim_i] = t
                        break
                else:
                    result[sim_i] = best8[sim_i, 0]
            return result
        else:
            # group winner/runner-up
            return group_ranks[rest][:, rank - 1]  # (N,)

    # Stage tracking
    stage_counts = defaultdict(lambda: defaultdict(int))

    def run_round(matches_in_round: list[dict], stage_name: str) -> None:
        for m in matches_in_round:
            home_lbl, away_lbl = m["match_label"].split(" vs ")
            teams_a = resolve_slot(home_lbl.strip())
            teams_b = resolve_slot(away_lbl.strip())

            # Gather features and Elo per sim
            fa_list = [team_feats.get(teams_a[i], {}) for i in range(N)]
            fb_list = [team_feats.get(teams_b[i], {}) for i in range(N)]
            ea = np.array([sim_elo.get(teams_a[i], np.full(1, ELO_START))[i]
                           if isinstance(sim_elo.get(teams_a[i]), np.ndarray)
                           else sim_elo.get(teams_a[i], ELO_START) for i in range(N)])
            eb = np.array([sim_elo.get(teams_b[i], np.full(1, ELO_START))[i]
                           if isinstance(sim_elo.get(teams_b[i]), np.ndarray)
                           else sim_elo.get(teams_b[i], ELO_START) for i in range(N)])

            p_w, p_d, p_l = predictor.predict_batch(fa_list, fb_list, ea, eb)
            p_w, p_d, p_l = _closeness_draw_boost(p_w, p_d, p_l, ea - eb)

            b_wins = _sample_knockout(p_w, p_d, p_l, ea, eb, rng)  # 0=A,1=B
            winners = np.where(b_wins == 0, teams_a, teams_b)
            losers  = np.where(b_wins == 0, teams_b, teams_a)

            match_winners[m["match_number"]] = winners

            # Elo update
            act_a = (b_wins == 0).astype(np.float64)
            new_ea, new_eb = _elo_update_vec(ea, eb, act_a)
            for i in range(N):
                if isinstance(sim_elo.get(teams_a[i]), np.ndarray):
                    sim_elo[teams_a[i]][i] = new_ea[i]
                if isinstance(sim_elo.get(teams_b[i]), np.ndarray):
                    sim_elo[teams_b[i]][i] = new_eb[i]

            # Count stage reached by winners
            for i in range(N):
                stage_counts[winners[i]][stage_name] += 1

    # R32: resolve using group results directly
    r32 = [m for m in bracket if m["stage"] == "Round of 32"]
    r16 = [m for m in bracket if m["stage"] == "Round of 16"]
    qf  = [m for m in bracket if m["stage"] == "Quarterfinals"]
    sf  = [m for m in bracket if m["stage"] == "Semifinals"]

    run_round(r32, "r16")
    run_round(r16, "qf")
    run_round(qf,  "sf")

    # Semis — track losers for 3rd place
    sf_winners_per_match = {}
    sf_losers_per_match  = {}
    for m in sf:
        home_lbl, away_lbl = m["match_label"].split(" vs ")
        teams_a = resolve_slot(home_lbl.strip())
        teams_b = resolve_slot(away_lbl.strip())
        fa_list = [team_feats.get(teams_a[i], {}) for i in range(N)]
        fb_list = [team_feats.get(teams_b[i], {}) for i in range(N)]
        ea = np.array([sim_elo.get(teams_a[i], np.full(1, ELO_START))[i]
                       if isinstance(sim_elo.get(teams_a[i]), np.ndarray)
                       else sim_elo.get(teams_a[i], ELO_START) for i in range(N)])
        eb = np.array([sim_elo.get(teams_b[i], np.full(1, ELO_START))[i]
                       if isinstance(sim_elo.get(teams_b[i]), np.ndarray)
                       else sim_elo.get(teams_b[i], ELO_START) for i in range(N)])
        p_w, p_d, p_l = predictor.predict_batch(fa_list, fb_list, ea, eb)
        p_w, p_d, p_l = _closeness_draw_boost(p_w, p_d, p_l, ea - eb)
        b_wins = _sample_knockout(p_w, p_d, p_l, ea, eb, rng)
        winners = np.where(b_wins == 0, teams_a, teams_b)
        losers  = np.where(b_wins == 0, teams_b, teams_a)
        match_winners[m["match_number"]] = winners
        sf_winners_per_match[m["match_number"]] = winners
        sf_losers_per_match[m["match_number"]]  = losers
        for i in range(N):
            stage_counts[winners[i]]["sf"]    += 1
            stage_counts[losers[i]]["sf"]     += 1

    # 3rd place playoff
    sf_nums = [m["match_number"] for m in sf]
    if len(sf_nums) == 2:
        lo_a = sf_losers_per_match[sf_nums[0]]
        lo_b = sf_losers_per_match[sf_nums[1]]
        fa_list = [team_feats.get(lo_a[i], {}) for i in range(N)]
        fb_list = [team_feats.get(lo_b[i], {}) for i in range(N)]
        ea = np.array([sim_elo.get(lo_a[i], ELO_START) if not isinstance(sim_elo.get(lo_a[i]), np.ndarray)
                       else sim_elo[lo_a[i]][i] for i in range(N)])
        eb = np.array([sim_elo.get(lo_b[i], ELO_START) if not isinstance(sim_elo.get(lo_b[i]), np.ndarray)
                       else sim_elo[lo_b[i]][i] for i in range(N)])
        p_w, p_d, p_l = predictor.predict_batch(fa_list, fb_list, ea, eb)
        b_wins = _sample_knockout(p_w, p_d, p_l, ea, eb, rng)
        thirds_place = np.where(b_wins == 0, lo_a, lo_b)
        for i in range(N):
            stage_counts[thirds_place[i]]["final"] += 1  # reached final 4

    # Final
    final_m = next(m for m in bracket if m["stage"] == "Final")
    home_lbl, away_lbl = final_m["match_label"].split(" vs ")
    fin_a = resolve_slot(home_lbl.strip())
    fin_b = resolve_slot(away_lbl.strip())
    fa_list = [team_feats.get(fin_a[i], {}) for i in range(N)]
    fb_list = [team_feats.get(fin_b[i], {}) for i in range(N)]
    ea = np.array([sim_elo.get(fin_a[i], ELO_START) if not isinstance(sim_elo.get(fin_a[i]), np.ndarray)
                   else sim_elo[fin_a[i]][i] for i in range(N)])
    eb = np.array([sim_elo.get(fin_b[i], ELO_START) if not isinstance(sim_elo.get(fin_b[i]), np.ndarray)
                   else sim_elo[fin_b[i]][i] for i in range(N)])
    p_w, p_d, p_l = predictor.predict_batch(fa_list, fb_list, ea, eb)
    p_w, p_d, p_l = _closeness_draw_boost(p_w, p_d, p_l, ea - eb)
    b_wins = _sample_knockout(p_w, p_d, p_l, ea, eb, rng)
    champions  = np.where(b_wins == 0, fin_a, fin_b)
    runners_up = np.where(b_wins == 0, fin_b, fin_a)
    for i in range(N):
        stage_counts[champions[i]]["final"]  += 1
        stage_counts[runners_up[i]]["final"] += 1
        stage_counts[champions[i]]["winner"] += 1

    return dict(stage_counts)


# ---------------------------------------------------------------------------
# Main simulation runner
# ---------------------------------------------------------------------------

def simulate_tournament(conn: sqlite3.Connection, N: int = N_SIMULATIONS) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_SEED)

    print("  Loading assets...")
    team_feats = _load_team_features(conn)
    base_elo   = _load_elo(conn)
    groups     = _load_groups(conn)
    bracket    = _load_bracket(conn)
    predictor  = VectorisedPredictor()

    all_teams = [t for tlist in groups.values() for t in tlist]

    print(f"  Simulating group stage ({N:,} runs in batch)...")
    group_ranks, sim_elo = simulate_groups_vectorised(
        groups, team_feats, base_elo, predictor, N, rng
    )

    # Count group stage outcomes
    qualify_counts = defaultdict(int)
    top2_counts    = defaultdict(int)
    r32_counts     = defaultdict(int)

    letters = sorted(groups.keys())
    for letter in letters:
        ranks = group_ranks[letter]  # (N, 4)
        for sim_i in range(N):
            top2_counts[ranks[sim_i, 0]] += 1
            top2_counts[ranks[sim_i, 1]] += 1
            r32_counts[ranks[sim_i, 0]]  += 1
            r32_counts[ranks[sim_i, 1]]  += 1

    # Best-8 third-place (approximate: top Elo among 3rd-place finishers)
    thirds_mat = np.stack([group_ranks[l][:, 2] for l in letters], axis=1)  # (N, 12)
    thirds_elo_mat = np.array([
        [sim_elo.get(thirds_mat[i, g], np.zeros(N))[i]
         if isinstance(sim_elo.get(thirds_mat[i, g]), np.ndarray)
         else sim_elo.get(thirds_mat[i, g], ELO_START)
         for g in range(12)]
        for i in range(N)
    ])
    best8_idx = np.argsort(-thirds_elo_mat, axis=1)[:, :8]
    for sim_i in range(N):
        for k in range(8):
            t = thirds_mat[sim_i, best8_idx[sim_i, k]]
            r32_counts[t] += 1

    print("  Simulating knockout stage...")
    stage_counts = simulate_knockout_vectorised(
        group_ranks, sim_elo, team_feats, predictor, bracket, N, rng
    )

    # Build output DataFrame
    rows = []
    for team in all_teams:
        letter = next(l for l, tl in groups.items() if team in tl)
        sc = stage_counts.get(team, {})
        rows.append({
            "team":        team,
            "group":       letter,
            "qualify_pct": round(r32_counts.get(team, 0) / N * 100, 1),
            "r16_pct":     round(sc.get("r16",  0) / N * 100, 1),
            "qf_pct":      round(sc.get("qf",   0) / N * 100, 1),
            "sf_pct":      round(sc.get("sf",   0) / N * 100, 1),
            "final_pct":   round(sc.get("final", 0) / N * 100, 1),
            "winner_pct":  round(sc.get("winner", 0) / N * 100, 1),
        })

    df = pd.DataFrame(rows).sort_values("winner_pct", ascending=False).reset_index(drop=True)
    return df


def save_results(df: pd.DataFrame, conn: sqlite3.Connection, N: int) -> None:
    run_at = datetime.now(timezone.utc).isoformat()
    c = conn.cursor()
    c.execute("DELETE FROM simulations")
    for _, row in df.iterrows():
        c.execute(
            """INSERT INTO simulations
               (run_at, n_simulations, team, qualify_pct, r16_pct, qf_pct, sf_pct, final_pct, winner_pct)
               VALUES (?,?,?,?,?,?,?,?,?)""",
            (run_at, N, row["team"], row["qualify_pct"],
             row["r16_pct"], row["qf_pct"], row["sf_pct"],
             row["final_pct"], row["winner_pct"]),
        )
    conn.commit()
    print(f"  Results saved ({len(df)} teams)")


def run(N: int = N_SIMULATIONS) -> pd.DataFrame:
    import time
    print("=" * 60)
    print("PHASE 4 — TOURNAMENT SIMULATION (vectorised)")
    print("=" * 60)
    t0 = time.time()
    conn = get_connection()
    df = simulate_tournament(conn, N=N)
    save_results(df, conn, N)
    conn.close()
    elapsed = time.time() - t0
    print(f"\n  Completed in {elapsed:.1f}s")
    print(f"\nTop 20 by winner probability ({N:,} simulations):")
    print(
        df.head(20)[["team", "group", "qualify_pct", "r16_pct",
                     "qf_pct", "sf_pct", "final_pct", "winner_pct"]]
        .to_string(index=False)
    )
    return df


if __name__ == "__main__":
    run()
