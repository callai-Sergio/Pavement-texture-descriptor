"""
texturelab_llm.logging_setup – Logger configuration for LLM calls.

Logs are written to ``logs/llm.log`` relative to the working directory.
The ``logs/`` directory is created automatically if it does not exist.
"""
import logging
import os
from pathlib import Path

_LOGGER_NAME = "texturelab_llm"
_LOG_DIR = Path("logs")
_LOG_FILE = _LOG_DIR / "llm.log"
_configured = False


def get_logger() -> logging.Logger:
    """Return (and lazily configure) the package-wide logger."""
    global _configured
    logger = logging.getLogger(_LOGGER_NAME)

    if not _configured:
        logger.setLevel(logging.DEBUG)

        # File handler – INFO and above
        try:
            _LOG_DIR.mkdir(parents=True, exist_ok=True)
            fh = logging.FileHandler(str(_LOG_FILE), encoding="utf-8")
            fh.setLevel(logging.INFO)
            fh.setFormatter(logging.Formatter(
                "%(asctime)s | %(levelname)-7s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            ))
            logger.addHandler(fh)
        except OSError:
            # If we cannot write logs (e.g. read-only FS), just skip file logging
            pass

        # Avoid duplicate handlers on repeated imports
        _configured = True

    return logger
