"""
Centralized logger for Exam Stress Balancer MLOps project.

Usage:
    from src.utils.logger import get_logger
    logger = get_logger(__name__)
    logger.info("Training started")
"""

import logging
import sys
from pathlib import Path
from logging.handlers import RotatingFileHandler

# Import config safely (avoid circular imports)
try:
    from src.config import LOG_FILE, LOG_LEVEL, LOG_FORMAT, LOG_DATE_FORMAT
except ImportError:
    from config import LOG_FILE, LOG_LEVEL, LOG_FORMAT, LOG_DATE_FORMAT  # type: ignore


_loggers: dict[str, logging.Logger] = {}


def get_logger(name: str = "exam_stress_balancer") -> logging.Logger:
    """
    Returns a named logger with both console and rotating file handlers.

    Args:
        name: Logger name (typically ``__name__`` of the calling module).

    Returns:
        Configured :class:`logging.Logger` instance.
    """
    if name in _loggers:
        return _loggers[name]

    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, LOG_LEVEL.upper(), logging.INFO))

    # Avoid duplicate handlers when called multiple times
    if logger.handlers:
        _loggers[name] = logger
        return logger

    formatter = logging.Formatter(fmt=LOG_FORMAT, datefmt=LOG_DATE_FORMAT)

    # ── Console handler ───────────────────────────────────────────────────────
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.DEBUG)
    logger.addHandler(console_handler)

    # ── Rotating file handler ─────────────────────────────────────────────────
    try:
        log_path = Path(LOG_FILE)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            log_path,
            maxBytes=10 * 1024 * 1024,   # 10 MB
            backupCount=5,
            encoding="utf-8",
        )
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.DEBUG)
        logger.addHandler(file_handler)
    except Exception as exc:  # pragma: no cover
        logger.warning("Could not create file handler: %s", exc)

    logger.propagate = False
    _loggers[name] = logger
    return logger


# Module-level convenience logger
log = get_logger("exam_stress_balancer")
