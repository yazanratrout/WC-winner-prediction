"""
Stakes-aware draw probability calibration.

Used during tournament simulation (not training) to adjust base W/D/L
probabilities based on:

  1. Team closeness  — if Elo diff is small, teams are evenly matched
                       → draw probability gets a bonus
  2. Matchday        — matchday 1 and 2 are neutral; matchday 3 teams
                       may play strategically based on what they need
  3. Qualification stakes — derived from current group standings:
       - Both qualified / both eliminated → "dead rubber": draw more likely
       - One team needs a win, other is safe → aggressor plays for win
       - One team needs a draw or win → draw becomes a rational target
       - One team needs a win but opponent also needs win → open game

Usage:
    from src.models.draw_calibration import adjust_for_stakes
    probs = adjust_for_stakes(base_probs, elo_diff, matchday, team_a_status, team_b_status)
    # probs: {"W": float, "D": float, "L": float} — sums to 1.0
"""

from __future__ import annotations

import numpy as np

# ── Closeness thresholds ──────────────────────────────────────────────────────
# Teams within ELO_CLOSE_THRESHOLD points are considered "evenly matched"
ELO_CLOSE_THRESHOLD = 100   # roughly top-30 vs top-30
ELO_VERY_CLOSE      = 50    # essentially equivalent teams

# How much draw probability gets boosted for close teams (additive share)
CLOSE_DRAW_BONUS      = 0.05
VERY_CLOSE_DRAW_BONUS = 0.10

# ── Matchday 3 stakes labels ─────────────────────────────────────────────────
# Passed in by the simulation engine based on current group standings.
STAKES = {
    "must_win":      "must_win",       # needs 3 points, draw not enough
    "draw_enough":   "draw_enough",    # needs at least 1 point
    "safe":          "safe",           # already qualified regardless
    "eliminated":    "eliminated",     # already out regardless
    "neutral":       "neutral",        # matchday 1 or 2, or unknown
}


def _normalise(probs: dict[str, float]) -> dict[str, float]:
    total = sum(probs.values())
    return {k: v / total for k, v in probs.items()}


def adjust_for_stakes(
    base_probs: dict[str, float],
    elo_diff: float,
    matchday: int = 1,
    team_a_status: str = "neutral",
    team_b_status: str = "neutral",
) -> dict[str, float]:
    """
    Adjust W/D/L probabilities for match context.

    Parameters
    ----------
    base_probs     : {"W": p_w, "D": p_d, "L": p_l} from the model, sum=1
    elo_diff       : home_elo - away_elo (positive = home team stronger)
    matchday       : 1, 2, or 3
    team_a_status  : qualification status of home team (STAKES values)
    team_b_status  : qualification status of away team (STAKES values)

    Returns
    -------
    Adjusted {"W": ..., "D": ..., "L": ...} summing to 1.0
    """
    p = dict(base_probs)  # copy

    # ── 1. Closeness boost ───────────────────────────────────────────────────
    abs_diff = abs(elo_diff)
    if abs_diff <= ELO_VERY_CLOSE:
        bonus = VERY_CLOSE_DRAW_BONUS
    elif abs_diff <= ELO_CLOSE_THRESHOLD:
        # Linear interpolation between the two thresholds
        t = (abs_diff - ELO_VERY_CLOSE) / (ELO_CLOSE_THRESHOLD - ELO_VERY_CLOSE)
        bonus = VERY_CLOSE_DRAW_BONUS * (1 - t) + CLOSE_DRAW_BONUS * t
    else:
        bonus = 0.0

    if bonus > 0:
        # Take the bonus equally from W and L proportional to their weight
        w_share = p["W"] / (p["W"] + p["L"] + 1e-9)
        p["W"] -= bonus * w_share
        p["L"] -= bonus * (1 - w_share)
        p["D"] += bonus

    # ── 2. Matchday 3 stakes adjustment ─────────────────────────────────────
    if matchday == 3:
        p = _apply_stakes(p, team_a_status, team_b_status)

    return _normalise({k: max(v, 0.01) for k, v in p.items()})


def _apply_stakes(
    p: dict[str, float],
    a_status: str,
    b_status: str,
) -> dict[str, float]:
    """
    Modify probabilities for matchday-3 qualification dynamics.

    The adjustments are additive shifts; we normalise afterwards.
    Positive shift on W means home team plays more aggressively.
    """
    a, b = a_status, b_status

    # Dead rubber — both teams have nothing to play for
    # → Both tend to rest players or play safe → draw more likely
    if a in ("safe", "eliminated") and b in ("safe", "eliminated"):
        p["D"] += 0.12
        p["W"] -= 0.06
        p["L"] -= 0.06
        return p

    # Home team safe, away team must win → away plays aggressively
    # (from home perspective: L more likely, W less likely)
    if a == "safe" and b == "must_win":
        p["L"] += 0.08
        p["W"] -= 0.08
        return p

    # Home team must win, away team safe → home plays aggressively
    if a == "must_win" and b == "safe":
        p["W"] += 0.08
        p["L"] -= 0.08
        return p

    # Home team safe, away team draw_enough → away plays cautiously
    # → Draw very attractive for away; home has no reason to push
    if a == "safe" and b == "draw_enough":
        p["D"] += 0.10
        p["W"] -= 0.05
        p["L"] -= 0.05
        return p

    # Home team draw_enough, away team safe
    if a == "draw_enough" and b == "safe":
        p["D"] += 0.10
        p["W"] -= 0.05
        p["L"] -= 0.05
        return p

    # Both teams need a draw or better → both play safe → draw likely
    if a == "draw_enough" and b == "draw_enough":
        p["D"] += 0.15
        p["W"] -= 0.075
        p["L"] -= 0.075
        return p

    # Both teams must win → open attacking game → draw slightly less likely
    if a == "must_win" and b == "must_win":
        p["D"] -= 0.06
        p["W"] += 0.03
        p["L"] += 0.03
        return p

    # Home must win, away draw_enough → tension: away defends, home attacks
    # → More likely: W or D (home scores once, away happy)
    if a == "must_win" and b == "draw_enough":
        p["W"] += 0.05
        p["D"] += 0.03
        p["L"] -= 0.08
        return p

    # Home draw_enough, away must win → reversed
    if a == "draw_enough" and b == "must_win":
        p["L"] += 0.05
        p["D"] += 0.03
        p["W"] -= 0.08
        return p

    # Home eliminated, away draw_enough → away likely to get what they need
    if a == "eliminated" and b == "draw_enough":
        p["D"] += 0.08
        p["L"] += 0.04
        p["W"] -= 0.12
        return p

    if a == "draw_enough" and b == "eliminated":
        p["D"] += 0.08
        p["W"] += 0.04
        p["L"] -= 0.12
        return p

    return p


def derive_team_status(
    team: str,
    standings: dict[str, dict],
    matches_played: int,
    remaining_matches: int = 1,
) -> str:
    """
    Derive a team's qualification status from current group standings.

    Parameters
    ----------
    team             : team name
    standings        : {team: {"pts": int, "gd": int, "gf": int}} for all 4 group teams
    matches_played   : how many matches this team has played so far
    remaining_matches: how many remain (usually 1 for matchday 3)

    Returns one of the STAKES labels.
    """
    if matches_played < 2:
        return "neutral"

    all_teams = sorted(standings.keys(),
                       key=lambda t: (-standings[t]["pts"], -standings[t]["gd"],
                                      -standings[t]["gf"]))

    team_pts = standings[team]["pts"]
    rank = all_teams.index(team) + 1  # 1-indexed

    # Points needed to guarantee top-2 (qualify directly from group of 4)
    # Maximum points 2nd-place team can reach
    pts_above = [standings[t]["pts"] for t in all_teams[:2] if t != team]
    pts_below = [standings[t]["pts"] for t in all_teams[2:] if t != team]

    max_pts_for_team = team_pts + 3 * remaining_matches

    # If already qualified (rank ≤ 2 and can't be overtaken by enough teams)
    # Simplified: if in top 2 with enough gap that 3rd can't catch up
    if rank <= 2:
        third_max = standings[all_teams[2]]["pts"] + 3
        if team_pts > third_max:
            return "safe"
        elif team_pts + 0 >= third_max:  # draw keeps us safe
            return "draw_enough"
        else:
            return "must_win" if rank == 2 else "draw_enough"
    else:
        # Outside top 2 — need to gain points
        second_pts = standings[all_teams[1]]["pts"]
        if max_pts_for_team < second_pts:
            return "eliminated"
        elif max_pts_for_team == second_pts:
            return "draw_enough"
        else:
            return "must_win"
