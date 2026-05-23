"""SQLite database setup — creates all tables if they don't exist."""

import os
import sqlite3
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

DB_PATH = os.getenv("DB_PATH", "data/db/wc_prediction.db")


def get_connection() -> sqlite3.Connection:
    Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db() -> None:
    conn = get_connection()
    c = conn.cursor()

    c.executescript("""
        CREATE TABLE IF NOT EXISTS teams (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            name        TEXT UNIQUE NOT NULL,
            fifa_rank   INTEGER,
            elo         REAL DEFAULT 1500.0,
            confederation TEXT,
            updated_at  TEXT
        );

        CREATE TABLE IF NOT EXISTS matches (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            date            TEXT NOT NULL,
            home_team       TEXT NOT NULL,
            away_team       TEXT NOT NULL,
            home_score      INTEGER,
            away_score      INTEGER,
            competition     TEXT,
            neutral         INTEGER DEFAULT 0,
            home_elo_before REAL,
            away_elo_before REAL,
            outcome         TEXT,
            source          TEXT,
            FOREIGN KEY (home_team) REFERENCES teams(name),
            FOREIGN KEY (away_team) REFERENCES teams(name)
        );

        CREATE TABLE IF NOT EXISTS players (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            name            TEXT NOT NULL,
            team            TEXT NOT NULL,
            position        TEXT,
            rating          REAL,
            market_value_m  REAL,
            age             INTEGER,
            caps            INTEGER DEFAULT 0,
            injured         INTEGER DEFAULT 0,
            club            TEXT,
            season          TEXT,
            FOREIGN KEY (team) REFERENCES teams(name)
        );

        CREATE TABLE IF NOT EXISTS simulations (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            run_at          TEXT NOT NULL,
            n_simulations   INTEGER,
            team            TEXT NOT NULL,
            group_win_pct   REAL,
            qualify_pct     REAL,
            r16_pct         REAL,
            qf_pct          REAL,
            sf_pct          REAL,
            final_pct       REAL,
            winner_pct      REAL,
            FOREIGN KEY (team) REFERENCES teams(name)
        );

        CREATE TABLE IF NOT EXISTS probability_history (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp   TEXT NOT NULL,
            team        TEXT NOT NULL,
            stage       TEXT NOT NULL,
            probability REAL NOT NULL,
            FOREIGN KEY (team) REFERENCES teams(name)
        );

        CREATE TABLE IF NOT EXISTS elo_history (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            date      TEXT NOT NULL,
            team      TEXT NOT NULL,
            elo       REAL NOT NULL,
            FOREIGN KEY (team) REFERENCES teams(name)
        );

        CREATE TABLE IF NOT EXISTS fifa_rankings (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            date      TEXT NOT NULL,
            team      TEXT NOT NULL,
            rank      INTEGER NOT NULL,
            points    REAL
        );
    """)

    conn.commit()
    conn.close()
    print(f"Database initialised at {DB_PATH}")


if __name__ == "__main__":
    init_db()
