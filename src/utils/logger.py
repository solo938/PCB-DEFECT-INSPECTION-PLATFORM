# src/utils/logger.py
"""
Centralized logging configuration for the entire project.

Usage:
    from src.utils.logger import get_logger
    logger = get_logger(__name__)
    logger.info("Hello world")
"""

import logging
import sys
from pathlib import Path

# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────

LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


# ─────────────────────────────────────────────
# Setup functions
# ─────────────────────────────────────────────

def setup_logging(log_file: Path = None, level: int = logging.INFO) -> None:
    """
    Configure logging for the entire application.

    Args:
        log_file: Path to log file. If None, only console logging.
        level: Logging level (default: logging.INFO)

    Notes:
        - Call this ONCE at application startup (in main.py or pipeline.py)
        - Avoids duplicate handlers when called multiple times
    """
    root_logger = logging.getLogger()

    # Remove existing handlers to avoid duplicates
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Set level
    root_logger.setLevel(level)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_formatter = logging.Formatter(LOG_FORMAT, DATE_FORMAT)
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)

    # File handler (optional)
    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(level)
        file_formatter = logging.Formatter(LOG_FORMAT, DATE_FORMAT)
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for a module.

    Args:
        name: Usually __name__ from the calling module

    Returns:
        logging.Logger: Configured logger instance
    """
    return logging.getLogger(name)