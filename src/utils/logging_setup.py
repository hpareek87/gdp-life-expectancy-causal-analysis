"""Centralized logging configuration."""
from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path

from .config import LOG_DIR


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Return a logger writing to both stdout and a per-run logfile."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(level)
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    log_path: Path = LOG_DIR / f"{datetime.now():%Y%m%d}_{name}.log"
    fh = logging.FileHandler(log_path)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    logger.propagate = False
    return logger
