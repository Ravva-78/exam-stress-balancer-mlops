#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# scripts/evaluate.sh — Run the RL evaluation pipeline
# ─────────────────────────────────────────────────────────────────────────────
# Usage:
#   bash scripts/evaluate.sh
#   EVAL_EPISODES=200 bash scripts/evaluate.sh
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
info()  { echo -e "${GREEN}[INFO]${NC}  $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*" >&2; }

info "Exam Stress Balancer — Evaluation Pipeline"

# ── Pre-flight checks ─────────────────────────────────────────────────────────
if [ ! -f "models/q_table.pkl" ]; then
    error "models/q_table.pkl not found. Run training first: bash scripts/train.sh"
    exit 1
fi

if ! command -v python &>/dev/null; then
    error "Python not found. Activate your virtual environment first."
    exit 1
fi

# ── Activate venv if present ──────────────────────────────────────────────────
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
elif [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
fi

export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"

export EVAL_EPISODES="${EVAL_EPISODES:-100}"
export EVAL_MAX_STEPS="${EVAL_MAX_STEPS:-200}"
export EVAL_SEED="${EVAL_SEED:-0}"

mkdir -p evaluations logs

info "Config → episodes=$EVAL_EPISODES | max_steps=$EVAL_MAX_STEPS | seed=$EVAL_SEED"

# ── Run evaluation ────────────────────────────────────────────────────────────
START_TIME=$(date +%s)
info "Starting evaluation at $(date)…"

python -m src.evaluation.evaluate_rl

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

# ── Verify output ─────────────────────────────────────────────────────────────
if [ -f "evaluations/evaluation_report.json" ]; then
    info "✔  Evaluation complete in ${ELAPSED}s"
    info "   Report → evaluations/evaluation_report.json"
    # Pretty-print key metrics
    if command -v python &>/dev/null; then
        python - <<'EOF'
import json, sys
try:
    with open("evaluations/evaluation_report.json") as f:
        r = json.load(f)
    agg = r.get("aggregate_metrics", {})
    print("\n  ── Aggregate Metrics ──")
    for k, v in agg.items():
        print(f"   {k:<22} {v}")
except Exception as e:
    print(f"  (could not parse report: {e})")
EOF
    fi
else
    error "Evaluation report not found — check logs/app.log for details."
    exit 1
fi
