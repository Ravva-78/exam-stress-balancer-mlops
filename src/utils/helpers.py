"""
Shared utility helpers for Exam Stress Balancer MLOps project.
"""

import json
import pickle
import hashlib
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.utils.logger import get_logger

logger = get_logger(__name__)


# ─── File I/O ─────────────────────────────────────────────────────────────────

def save_pickle(obj: Any, path: Path | str) -> None:
    """Serialize *obj* to *path* using pickle."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fh:
        pickle.dump(obj, fh)
    logger.info("Saved pickle → %s", path)


def load_pickle(path: Path | str) -> Any:
    """Deserialize and return the object stored at *path*."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Pickle file not found: {path}")
    with open(path, "rb") as fh:
        obj = pickle.load(fh)
    logger.info("Loaded pickle ← %s", path)
    return obj


def save_json(data: dict, path: Path | str, indent: int = 2) -> None:
    """Write *data* as pretty-printed JSON to *path*."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=indent, default=str)
    logger.info("Saved JSON → %s", path)


def load_json(path: Path | str) -> dict:
    """Load and return JSON from *path*."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    logger.info("Loaded JSON ← %s", path)
    return data


# ─── Metadata Helpers ─────────────────────────────────────────────────────────

def build_metadata(extra: dict | None = None) -> dict:
    """
    Build a standard metadata dictionary for a training run.

    Args:
        extra: Optional dict merged into the base metadata.

    Returns:
        Metadata dict with ``created_at``, ``version``, and ``checksum`` keys.
    """
    meta = {
        "project": "exam-stress-balancer",
        "version": "1.0.0",
        "created_at": utc_now_iso(),
        "framework": "custom-rl",
    }
    if extra:
        meta.update(extra)
    return meta


def utc_now_iso() -> str:
    """Return the current UTC timestamp in ISO-8601 format."""
    return datetime.now(timezone.utc).isoformat()


def file_checksum(path: Path | str, algorithm: str = "sha256") -> str:
    """
    Compute and return the hex-digest checksum of a file.

    Args:
        path:      Path to the file.
        algorithm: Hash algorithm name (default ``"sha256"``).

    Returns:
        Hex-digest string.
    """
    path = Path(path)
    h = hashlib.new(algorithm)
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


# ─── Timing ──────────────────────────────────────────────────────────────────

class Timer:
    """Context manager that measures wall-clock elapsed time in seconds."""

    def __init__(self, name: str = "block"):
        self.name = name
        self.elapsed: float = 0.0

    def __enter__(self) -> "Timer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, *_) -> None:
        self.elapsed = time.perf_counter() - self._start
        logger.debug("⏱  %s completed in %.3f s", self.name, self.elapsed)


# ─── Stress / Action Helpers ─────────────────────────────────────────────────

def stress_level_to_index(level: str) -> int:
    """Map a stress-level string to its integer index."""
    mapping = {"low": 0, "medium": 1, "high": 2, "critical": 3}
    level = level.lower().strip()
    if level not in mapping:
        raise ValueError(f"Unknown stress level '{level}'. Valid: {list(mapping)}")
    return mapping[level]


def index_to_action(index: int) -> str:
    """Map an action index to its human-readable label."""
    actions = ["rest", "light_study", "moderate_study", "intense_study"]
    if not (0 <= index < len(actions)):
        raise ValueError(f"Action index {index} out of range [0, {len(actions)-1}]")
    return actions[index]


def validate_state(state: dict) -> None:
    """
    Lightweight validation of an incoming state payload.

    Raises:
        ValueError: If required fields are missing or invalid.
    """
    required = ["stress_level", "hours_studied", "days_until_exam"]
    for field in required:
        if field not in state:
            raise ValueError(f"Missing required state field: '{field}'")

    if state["stress_level"] not in ["low", "medium", "high", "critical"]:
        raise ValueError(f"Invalid stress_level: '{state['stress_level']}'")

    if not (0 <= float(state["hours_studied"]) <= 24):
        raise ValueError("hours_studied must be between 0 and 24")

    if int(state["days_until_exam"]) < 0:
        raise ValueError("days_until_exam must be non-negative")
