#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# scripts/run_api.sh — Start the FastAPI prediction server
# ─────────────────────────────────────────────────────────────────────────────
# Usage:
#   bash scripts/run_api.sh              # production mode
#   API_RELOAD=true bash scripts/run_api.sh   # development mode with hot-reload
#   API_PORT=9000  bash scripts/run_api.sh    # custom port
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; CYAN='\033[0;36m'; NC='\033[0m'
info()  { echo -e "${GREEN}[INFO]${NC}  $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*" >&2; }

info "Exam Stress Balancer — FastAPI Server"

# ── Pre-flight checks ─────────────────────────────────────────────────────────
if ! command -v python &>/dev/null; then
    error "Python not found. Activate your virtual environment first."
    exit 1
fi

if ! python -c "import uvicorn" 2>/dev/null; then
    error "uvicorn not installed. Run: pip install -r requirements.txt"
    exit 1
fi

# ── Activate venv if present ──────────────────────────────────────────────────
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
    info "Virtual environment activated."
elif [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
    info "Virtual environment activated."
fi

# ── Environment ───────────────────────────────────────────────────────────────
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
export API_HOST="${API_HOST:-0.0.0.0}"
export API_PORT="${API_PORT:-8000}"
export API_WORKERS="${API_WORKERS:-1}"
export API_RELOAD="${API_RELOAD:-false}"
export LOG_LEVEL="${LOG_LEVEL:-info}"

mkdir -p models data evaluations logs

# ── Model check ───────────────────────────────────────────────────────────────
if [ ! -f "models/q_table.pkl" ]; then
    warn "No trained model found at models/q_table.pkl."
    warn "The API will start in degraded mode (503 on /predict)."
    warn "Run training first: bash scripts/train.sh"
fi

# ── Build uvicorn command ─────────────────────────────────────────────────────
UVICORN_ARGS="src.api.main:app --host $API_HOST --port $API_PORT --log-level $LOG_LEVEL"

if [ "$API_RELOAD" = "true" ]; then
    UVICORN_ARGS="$UVICORN_ARGS --reload"
    warn "Hot-reload enabled (development mode). Do not use in production."
else
    UVICORN_ARGS="$UVICORN_ARGS --workers $API_WORKERS"
fi

# ── Launch ────────────────────────────────────────────────────────────────────
echo ""
echo -e "  ${CYAN}┌────────────────────────────────────────────────┐${NC}"
echo -e "  ${CYAN}│  Exam Stress Balancer API                      │${NC}"
echo -e "  ${CYAN}│  http://$API_HOST:$API_PORT                         │${NC}"
echo -e "  ${CYAN}│  Docs → http://localhost:$API_PORT/docs           │${NC}"
echo -e "  ${CYAN}└────────────────────────────────────────────────┘${NC}"
echo ""

exec python -m uvicorn $UVICORN_ARGS
