"""
Live data fetcher using API-Football.

Free tier: 100 requests/day.
Register at https://dashboard.api-football.com and paste your key in .env as API_FOOTBALL_KEY.

Run:
  python -m src.data.api_client
"""

import os
import time
import sqlite3
from datetime import date, datetime

import requests
from dotenv import load_dotenv

from src.data.database import get_connection
from src.data.team_names import canonical

load_dotenv()

API_KEY = os.getenv("API_FOOTBALL_KEY", "")
BASE_URL = "https://v3.football.api-sports.io"
HEADERS = {
    "x-apisports-key": API_KEY,
}

# API-Football ID for international matches / World Cup
LEAGUE_IDS = {
    "world_cup": 1,
    "world_cup_qual_europe": 32,
    "world_cup_qual_south_america": 34,
    "world_cup_qual_africa": 29,
    "world_cup_qual_asia": 30,
    "world_cup_qual_north_america": 31,
    "world_cup_qual_oceania": 33,
    "nations_league": 5,
    "copa_america": 9,
    "euro": 4,
    "afcon": 6,
}


# ---------------------------------------------------------------------------
# Core request helper
# ---------------------------------------------------------------------------

def _get(endpoint: str, params: dict) -> dict:
    if not API_KEY or API_KEY == "your_key_here":
        raise RuntimeError("API_FOOTBALL_KEY not set in .env")
    resp = requests.get(f"{BASE_URL}/{endpoint}", headers=HEADERS, params=params, timeout=15)
    resp.raise_for_status()
    data = resp.json()
    if data.get("errors"):
        raise RuntimeError(f"API error: {data['errors']}")
    return data


# ---------------------------------------------------------------------------
# Fetch recent fixture results
# ---------------------------------------------------------------------------

def fetch_recent_fixtures(conn: sqlite3.Connection, league_id: int, season: int) -> int:
    """Fetch all finished fixtures for a league/season and store them."""
    data = _get("fixtures", {"league": league_id, "season": season, "status": "FT"})
    fixtures = data.get("response", [])

    c = conn.cursor()
    inserted = 0
    for f in fixtures:
        match_date = f["fixture"]["date"][:10]
        home = canonical(f["teams"]["home"]["name"])
        away = canonical(f["teams"]["away"]["name"])
        home_score = f["goals"]["home"]
        away_score = f["goals"]["away"]
        neutral = f["fixture"].get("venue", {}).get("city") is None

        if home_score is None or away_score is None:
            continue

        if home_score > away_score:
            outcome = "W"
        elif home_score == away_score:
            outcome = "D"
        else:
            outcome = "L"

        # Ensure teams exist
        for team in (home, away):
            c.execute("INSERT OR IGNORE INTO teams (name) VALUES (?)", (team,))

        try:
            c.execute(
                """
                INSERT INTO matches
                  (date, home_team, away_team, home_score, away_score,
                   competition, neutral, outcome, source)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (match_date, home, away, home_score, away_score,
                 str(league_id), int(neutral), outcome, "api-football"),
            )
            inserted += 1
        except sqlite3.IntegrityError:
            pass

        time.sleep(0.1)  # stay within rate limits

    conn.commit()
    print(f"  Fetched {inserted} new fixtures (league {league_id}, season {season})")
    return inserted


# ---------------------------------------------------------------------------
# Fetch live FIFA rankings
# ---------------------------------------------------------------------------

def fetch_fifa_rankings(conn: sqlite3.Connection) -> int:
    """Fetch current FIFA world rankings."""
    data = _get("standings", {"league": LEAGUE_IDS["world_cup"], "season": 2026})
    # FIFA rankings are not directly in standings — use the teams ranking endpoint
    # API-Football exposes rankings under /rankings/worlds
    data = _get("rankings/worlds", {})
    rankings = data.get("response", [])

    today = str(date.today())
    c = conn.cursor()
    inserted = 0
    for entry in rankings:
        team = canonical(entry.get("team", {}).get("name", ""))
        rank = entry.get("rank")
        points = entry.get("points")
        if not team or not rank:
            continue
        c.execute(
            "INSERT INTO fifa_rankings (date, team, rank, points) VALUES (?, ?, ?, ?)",
            (today, team, rank, points),
        )
        # Update current rank on teams table
        c.execute(
            "UPDATE teams SET fifa_rank = ?, updated_at = ? WHERE name = ?",
            (rank, today, team),
        )
        inserted += 1

    conn.commit()
    print(f"  Fetched {inserted} FIFA ranking entries")
    return inserted


# ---------------------------------------------------------------------------
# Fetch squad / injury data
# ---------------------------------------------------------------------------

def fetch_squad_injuries(conn: sqlite3.Connection, team_api_id: int, team_name: str) -> int:
    """Fetch active injuries for a national team and mark players as injured."""
    try:
        data = _get("injuries", {"team": team_api_id, "season": 2026})
    except Exception as e:
        print(f"  Could not fetch injuries for {team_name}: {e}")
        return 0

    injuries = data.get("response", [])
    c = conn.cursor()
    for inj in injuries:
        player_name = inj.get("player", {}).get("name", "")
        c.execute(
            "UPDATE players SET injured = 1 WHERE name = ? AND team = ?",
            (player_name, team_name),
        )

    conn.commit()
    return len(injuries)


# ---------------------------------------------------------------------------
# Entry point — run a quick sync for WC 2026 qualifiers
# ---------------------------------------------------------------------------

def run_sync(seasons: list[int] = None) -> None:
    if seasons is None:
        seasons = [2024, 2025, 2026]

    print("=== Live API sync ===")
    conn = get_connection()

    for league_name, league_id in LEAGUE_IDS.items():
        for season in seasons:
            try:
                fetch_recent_fixtures(conn, league_id, season)
            except RuntimeError as e:
                print(f"  {league_name} season {season}: {e}")
            time.sleep(0.5)

    try:
        fetch_fifa_rankings(conn)
    except RuntimeError as e:
        print(f"  FIFA rankings: {e}")

    conn.close()
    print("API sync complete.")


if __name__ == "__main__":
    run_sync()
