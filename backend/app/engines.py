"""Which engine each optional component actually runs with, and why a fallback happened.

Pipeline code records it here; the worker publishes the snapshot with its heartbeat so
the UI can show e.g. "asr_secondary: faster-whisper:base (IndicConformer: HF_TOKEN not set)".
"""
from __future__ import annotations

import threading

_status: dict[str, str] = {}
_lock = threading.Lock()


def set_status(component: str, value: str) -> None:
    with _lock:
        _status[component] = value


def snapshot() -> dict[str, str]:
    with _lock:
        return dict(_status)
