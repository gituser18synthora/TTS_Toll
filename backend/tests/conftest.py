"""Shared fixtures: an isolated DATA_ROOT + config per test session, synthetic audio."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

# hermetic: no model downloads, no network (model-backed paths have their own tests)
TEST_CONFIG = """
paths:
  data_root: ./data
  models_dir: ./models
device: cpu
vad:
  engine: energy
denoise:
  engine: none
quality:
  engine: snr
  tagger:
    enabled: false
asr:
  primary:
    engine: none
  secondary:
    engine: none
speaker_check:
  enabled: false
"""


@pytest.fixture(scope="session")
def portal_env(tmp_path_factory) -> Path:
    """Point the app at a throwaway config + data folder before anything imports app.main."""
    root = tmp_path_factory.mktemp("portal")
    (root / "config.yaml").write_text(TEST_CONFIG)
    os.environ["PORTAL_CONFIG"] = str(root / "config.yaml")
    os.environ.pop("PORTAL_DATA_ROOT", None)
    from app.config import get_settings

    get_settings.cache_clear()
    return root


@pytest.fixture(scope="session")
def client(portal_env):
    from fastapi.testclient import TestClient

    from app.main import create_app

    with TestClient(create_app()) as c:
        yield c


def speechlike(duration: float, sr: int, rng: np.random.Generator) -> np.ndarray:
    """Amplitude-modulated noise burst (syllable-rate envelope) — energy VAD treats it as speech."""
    t = np.arange(int(duration * sr)) / sr
    env = 0.55 + 0.45 * np.sin(2 * np.pi * 4.0 * t + rng.uniform(0, 6.28))
    return (0.2 * env * rng.standard_normal(len(t))).astype(np.float32)


def make_recording(path: Path, layout: list[tuple[str, float]], sr: int = 44100, seed: int = 0) -> list[tuple[float, float]]:
    """Write a wav from [("speech"|"silence", seconds), ...]; returns the speech spans (start, end)."""
    rng = np.random.default_rng(seed)
    parts, spans, t = [], [], 0.0
    for kind, dur in layout:
        if kind == "speech":
            parts.append(speechlike(dur, sr, rng))
            spans.append((t, t + dur))
        else:
            parts.append((0.0005 * rng.standard_normal(int(dur * sr))).astype(np.float32))
        t += dur
    sf.write(str(path), np.concatenate(parts), sr, subtype="PCM_16")
    return spans
