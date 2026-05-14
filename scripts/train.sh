#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# scripts/train.sh — Run the RL training pipeline
# ─────────────────────────────────────────────────────────────────────────────
# Usage:
#   bash scripts/train.sh                 # default config from params.yaml
#   TRAIN_EPISODES=1000 bash scripts/train.sh
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

# ── Resolve project root ──────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# ── Colour helpers ────────────────────────────────────────────────────────────
GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
info()  { echo -e "${GREEN}[INFO]${NC}  $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*" >&2; }

# ── Pre-flight checks ─────────────────────────────────────────────────────────
info "Exam Stress Balancer — Training Pipeline"
info "Project root: $PROJECT_ROOT"

if ! command -v python &>/dev/null; then
    error "Python not found. Activate your virtual environment first."
    exit 1
fi

PYTHON_VERSION=$(python --version 2>&1)
info "Python: $PYTHON_VERSION"

# ── Create required directories ───────────────────────────────────────────────
mkdir -p models data evaluations logs
info "Directories ready."

# ── Optional: activate venv if it exists ─────────────────────────────────────
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
    info "Virtual environment activated."
elif [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
    info "Virtual environment activated."
fi

# ── Export environment overrides (optional) ───────────────────────────────────
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
export MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI:-sqlite:///mlflow.db}"
export MLFLOW_EXPERIMENT_NAME="${MLFLOW_EXPERIMENT_NAME:-exam-stress-balancer}"

# Hyperparameter overrides (read from env if set)
export TRAIN_EPISODES="${TRAIN_EPISODES:-500}"
export TRAIN_LR="${TRAIN_LR:-0.1}"
export TRAIN_GAMMA="${TRAIN_GAMMA:-0.95}"
export TRAIN_EPSILON="${TRAIN_EPSILON:-1.0}"
export TRAIN_EPS_MIN="${TRAIN_EPS_MIN:-0.01}"
export TRAIN_EPS_DECAY="${TRAIN_EPS_DECAY:-0.995}"
export TRAIN_SEED="${TRAIN_SEED:-42}"

info "Config → episodes=$TRAIN_EPISODES | lr=$TRAIN_LR | gamma=$TRAIN_GAMMA"
info "MLflow → $MLFLOW_TRACKING_URI"

# ── Run training ──────────────────────────────────────────────────────────────
START_TIME=$(date +%s)
info "Starting training at $(date)…"

python -m src.training.train_rl

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

# ── Verify artefacts ──────────────────────────────────────────────────────────
if [ -f "models/q_table.pkl" ] && [ -f "models/metadata.json" ]; then
    info "✔  Training complete in ${ELAPSED}s"
    info "   models/q_table.pkl   ($(du -sh models/q_table.pkl   | cut -f1))"
    info "   models/metadata.json ($(du -sh models/metadata.json | cut -f1))"
else
    error "Training artefacts not found — check logs/app.log for details."
    exit 1
fi
