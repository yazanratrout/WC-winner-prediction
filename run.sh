#!/usr/bin/env bash
set -e

# ── Colours ──────────────────────────────────────────────────────────────────
GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
info()  { echo -e "${GREEN}[setup]${NC} $*"; }
warn()  { echo -e "${YELLOW}[warn]${NC}  $*"; }
error() { echo -e "${RED}[error]${NC} $*"; exit 1; }

# ── 1. Check dependencies ────────────────────────────────────────────────────
command -v python3 >/dev/null 2>&1 || error "Python 3 is required. Install from https://python.org"
command -v node    >/dev/null 2>&1 || error "Node.js is required. Install from https://nodejs.org"
command -v npm     >/dev/null 2>&1 || error "npm is required (comes with Node.js)"

PY_VERSION=$(python3 -c 'import sys; print(sys.version_info.minor)')
[ "$PY_VERSION" -ge 10 ] || error "Python 3.10+ required (found 3.$PY_VERSION)"

info "Python $(python3 --version)  |  Node $(node --version)  |  npm $(npm --version)"

# ── 2. Python virtual environment ────────────────────────────────────────────
if [ ! -d "venv" ]; then
  info "Creating Python virtual environment..."
  python3 -m venv venv
fi

info "Activating venv and installing Python dependencies..."
source venv/bin/activate
pip install -q --upgrade pip
pip install -q -r requirements.txt
info "Python dependencies installed."

# ── 3. Frontend dependencies ─────────────────────────────────────────────────
if [ ! -d "frontend/node_modules" ]; then
  info "Installing frontend dependencies (npm install)..."
  npm --prefix frontend install --silent
  info "Frontend dependencies installed."
else
  info "Frontend node_modules already present, skipping npm install."
fi

# ── 4. Start API ─────────────────────────────────────────────────────────────
info "Starting FastAPI backend on http://localhost:8000 ..."
source venv/bin/activate
uvicorn api.main:app --port 8000 --log-level warning &
API_PID=$!

# Wait for the API to be ready (up to 20s)
echo -n "Waiting for API"
for i in $(seq 1 20); do
  sleep 1
  if curl -s http://localhost:8000/health >/dev/null 2>&1; then
    echo ""
    info "API is up."
    break
  fi
  echo -n "."
  if [ "$i" -eq 20 ]; then
    echo ""
    error "API did not start in time. Check for port conflicts on 8000."
  fi
done

# ── 5. Start frontend ─────────────────────────────────────────────────────────
info "Starting frontend on http://localhost:5173 ..."
npm --prefix frontend run dev &
FRONTEND_PID=$!

echo ""
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}  WC 2026 Prediction Engine is running!${NC}"
echo -e "${GREEN}  Dashboard → http://localhost:5173${NC}"
echo -e "${GREEN}  API docs  → http://localhost:8000/docs${NC}"
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "  Press ${YELLOW}Ctrl+C${NC} to stop both servers."
echo ""

# ── 6. Cleanup on exit ────────────────────────────────────────────────────────
trap "echo ''; info 'Stopping servers...'; kill $API_PID $FRONTEND_PID 2>/dev/null; exit 0" INT TERM

wait
