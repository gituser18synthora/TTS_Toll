"""Console + rotating file logging (DATA_ROOT/logs/<name>.log)."""
from __future__ import annotations

import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path

_configured = False

NOISY_LOGGERS = ("faster_whisper", "httpx", "httpcore", "urllib3", "huggingface_hub", "multipart",
                 "python_multipart")


def setup_logging(name: str, log_dir: Path) -> None:
    global _configured
    if _configured:
        return
    level = getattr(logging, os.environ.get("PORTAL_LOG_LEVEL", "INFO").upper(), logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(levelname)-7s [%(name)s] %(message)s", "%Y-%m-%d %H:%M:%S")

    root = logging.getLogger()
    root.setLevel(level)
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    root.addHandler(console)

    log_dir.mkdir(parents=True, exist_ok=True)
    file_handler = RotatingFileHandler(log_dir / f"{name}.log", maxBytes=10 * 1024 * 1024,
                                       backupCount=5, encoding="utf-8")
    file_handler.setFormatter(fmt)
    root.addHandler(file_handler)

    for noisy in NOISY_LOGGERS:
        logging.getLogger(noisy).setLevel(logging.WARNING)
    _configured = True
