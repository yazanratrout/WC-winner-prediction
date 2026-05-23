# WC 2026 Winner Prediction Engine

A full-stack machine learning project that simulates the 2026 FIFA World Cup 10,000 times to predict each of the 48 teams' probability of winning, reaching the final, semi-finals, and beyond.

---

## What it does

- **Match prediction** — XGBoost classifier trained on 11,700+ international matches (2014–2024), calibrated with isotonic regression. Features include Elo ratings, squad quality, recent form, xG, chemistry, and tactical proxies.
- **Tournament simulation** — Vectorised Monte Carlo engine runs 10,000 full tournaments (group stage → R32 → R16 → QF → SF → Final) in ~10 seconds. Includes in-tournament Elo updates, stakes-aware draw calibration, and penalty shootout simulation.
- **Live updating** — FastAPI backend with APScheduler polls for new match results and re-simulates after every real game.
- **React dashboard** — 5-page interactive UI with animated probability bars, country flags, head-to-head heatmaps, and a match predictor.

**Current top predictions:** Spain 16.0% · Argentina 15.7% · France 10.1%

---

## Quick start

**Requirements:** Python 3.10+, Node.js 18+

```bash
git clone https://github.com/yourusername/WC-winner-prediction.git
cd WC-winner-prediction
bash run.sh
```

The script will:

1. Create a Python virtual environment and install dependencies
2. Install frontend npm packages
3. Start the API on `http://localhost:8000`
4. Start the dashboard on `http://localhost:5173`

Press `Ctrl+C` to stop both servers.

---

## Manual setup

If you prefer to run things separately:

```bash
# Backend
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn api.main:app --port 8000 --reload

# Frontend (new terminal)
cd frontend
npm install
npm run dev
```

---

## Project structure

```text
├── api/                  # FastAPI backend (8 endpoints)
│   └── main.py
├── src/
│   ├── data/             # DB connection, API client, live updater
│   ├── features/         # Feature engineering pipeline
│   ├── models/           # Model training, draw calibration
│   └── simulation/       # Monte Carlo tournament engine
├── frontend/             # React + Vite dashboard
│   └── src/pages/        # Home, Groups, Bracket, Predict, TeamDetail
├── data/
│   ├── db/               # SQLite database (11,727 matches, 48 teams)
│   └── processed/        # features.parquet (43 features × 11,727 rows)
├── models/               # Trained XGBoost model (.pkl)
├── requirements.txt
└── run.sh                # One-command setup & launch
```

---

## API endpoints

| Method | Endpoint          | Description                                       |
| ------ | ----------------- | ------------------------------------------------- |
| GET    | `/health`         | Liveness check                                    |
| GET    | `/teams`          | All 48 teams sorted by win probability            |
| GET    | `/team/{name}`    | Team profile + probabilities + Elo history        |
| GET    | `/group/{letter}` | Group standings with probabilities                |
| POST   | `/match/predict`  | Predict W/D/L for any two teams                   |
| POST   | `/match/result`   | Submit a real result, updates Elo + re-simulates  |
| POST   | `/simulate`       | Trigger a fresh 10,000-run simulation             |
| GET    | `/history/{team}` | Probability evolution over time                   |

Interactive docs available at `http://localhost:8000/docs`

---

## Tech stack

**Backend:** Python · FastAPI · XGBoost · scikit-learn · SQLite · APScheduler  
**Frontend:** React · Vite · Recharts · React Router · CSS Modules  
**ML:** XGBoost multiclass classifier · isotonic calibration · Monte Carlo simulation
