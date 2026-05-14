"""
Central configuration for Exam Stress Balancer MLOps project.
All paths, hyperparameters, and environment settings live here.
"""

import os
from pathlib import Path

# ─── Project Root ────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ─── Directory Paths ─────────────────────────────────────────────────────────
MODELS_DIR       = PROJECT_ROOT / "models"
DATA_DIR         = PROJECT_ROOT / "data"
EVALUATIONS_DIR  = PROJECT_ROOT / "evaluations"
LOGS_DIR         = PROJECT_ROOT / "logs"
RESULTS_DIR      = PROJECT_ROOT / "results"

# Ensure directories exist at import time
for _dir in [MODELS_DIR, DATA_DIR, EVALUATIONS_DIR, LOGS_DIR, RESULTS_DIR]:
    _dir.mkdir(parents=True, exist_ok=True)

# ─── Model Artifact Paths ────────────────────────────────────────────────────
Q_TABLE_PATH       = MODELS_DIR / "q_table.pkl"
MODEL_METADATA_PATH = MODELS_DIR / "metadata.json"

# ─── Evaluation Output ───────────────────────────────────────────────────────
EVALUATION_REPORT_PATH = EVALUATIONS_DIR / "evaluation_report.json"

# ─── Logging ─────────────────────────────────────────────────────────────────
LOG_FILE        = LOGS_DIR / "app.log"
LOG_LEVEL       = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT      = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# ─── MLflow ──────────────────────────────────────────────────────────────────
import pathlib
_mlruns = pathlib.Path(__file__).parent.parent / "mlruns"
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "file:///" + str(_mlruns).replace("\\", "/"))
MLFLOW_EXPERIMENT_NAME  = os.getenv("MLFLOW_EXPERIMENT_NAME", "exam-stress-balancer")

# ─── Training Hyperparameters (override with env vars or RL logic) ────────────
TRAINING_CONFIG = {
    "episodes":       int(os.getenv("TRAIN_EPISODES",    "500")),
    "max_steps":      int(os.getenv("TRAIN_MAX_STEPS",   "200")),
    "learning_rate":  float(os.getenv("TRAIN_LR",        "0.1")),
    "discount_factor": float(os.getenv("TRAIN_GAMMA",    "0.95")),
    "epsilon":        float(os.getenv("TRAIN_EPSILON",   "1.0")),
    "epsilon_min":    float(os.getenv("TRAIN_EPS_MIN",   "0.01")),
    "epsilon_decay":  float(os.getenv("TRAIN_EPS_DECAY", "0.995")),
    "seed":           int(os.getenv("TRAIN_SEED",        "42")),
    "algorithm": "q_learning",  # default
}

# ─── Evaluation Config ───────────────────────────────────────────────────────
EVALUATION_CONFIG = {
    "num_episodes": int(os.getenv("EVAL_EPISODES", "100")),
    "max_steps":    int(os.getenv("EVAL_MAX_STEPS", "200")),
    "seed":         int(os.getenv("EVAL_SEED",      "0")),
}

# ─── API Config ──────────────────────────────────────────────────────────────
API_HOST    = os.getenv("API_HOST", "0.0.0.0")
API_PORT    = int(os.getenv("API_PORT", "8000"))
API_WORKERS = int(os.getenv("API_WORKERS", "1"))
API_RELOAD  = os.getenv("API_RELOAD", "false").lower() == "true"

# ─── Environment / State Space ───────────────────────────────────────────────
# Placeholder — replace with your actual state/action definitions
ENV_CONFIG = {
    "stress_levels":    ["low", "medium", "high", "critical"],
    "actions":         ["rest", "light_study", "moderate_study", "intense_study"],
    "state_dim":        4,
    "action_dim":       4,
}
