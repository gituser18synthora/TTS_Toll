"""Real-model checks (DNSMOS, DeepFilterNet). Skipped when the weights are not in models/."""
from __future__ import annotations

import numpy as np
import pytest
import soundfile as sf

from app.config import PROJECT_ROOT
from app.devices import has_module

MODELS = PROJECT_ROOT / "models"
DNSMOS_READY = (MODELS / "dnsmos" / "sig_bak_ovr.onnx").exists() and has_module("onnxruntime")
DF_READY = (MODELS / "deepfilternet" / "DeepFilterNet3" / "config.ini").exists() and has_module("df")


def voiced(seconds: float, sr: int, gaps: bool = False) -> np.ndarray:
    """Harmonic 'vowel' with a pitch contour and syllable envelope; optional silent gaps."""
    t = np.arange(int(seconds * sr)) / sr
    f0 = 140 + 25 * np.sin(2 * np.pi * 0.4 * t)
    phase = 2 * np.pi * np.cumsum(f0) / sr
    x = sum(np.sin(k * phase) / k for k in range(1, 25)) * np.clip(np.sin(2 * np.pi * 2.5 * t), 0, None)
    if gaps:
        x[(t % 2.0) > 1.5] = 0.0  # 0.5 s of silence every 2 s
    return (0.08 * x).astype(np.float32)


@pytest.mark.skipif(not DNSMOS_READY, reason="DNSMOS model not downloaded")
def test_dnsmos_scores_and_noise_lowers_background_score():
    from app.pipeline.quality import Dnsmos

    dnsmos = Dnsmos(MODELS)
    clean = voiced(6.0, 16000)
    noisy = clean + 0.05 * np.random.default_rng(0).standard_normal(len(clean)).astype(np.float32)
    c, n = dnsmos.score(clean), dnsmos.score(noisy)
    assert c.method == "dnsmos" and set(c.details) == {"sig", "bak", "ovrl"}
    assert 1.0 <= n.score <= 5.0 and 1.0 <= c.score <= 5.0
    assert n.details["bak"] < c.details["bak"]


@pytest.mark.skipif(not DF_READY, reason="DeepFilterNet not installed / weights not downloaded")
def test_deepfilternet_removes_noise_in_pauses(tmp_path):
    from app.pipeline.denoise import DF_SR, DeepFilter, enhance_file

    sr = DF_SR
    clean = voiced(8.0, sr, gaps=True)
    noisy = clean + 0.01 * np.random.default_rng(1).standard_normal(len(clean)).astype(np.float32)
    src, out = tmp_path / "in.wav", tmp_path / "out.wav"
    sf.write(str(src), noisy, sr, subtype="FLOAT")
    enhance_file(DeepFilter(MODELS), src, out, chunk_sec=3.0, atten_lim_db=None)  # several chunk seams
    y, ysr = sf.read(str(out), dtype="float32")
    assert ysr == sr and len(y) == len(noisy)
    t = np.arange(len(y)) / sr
    gap = ((t % 2.0) > 1.6) & ((t % 2.0) < 1.95)  # well inside the silent gaps
    before = 10 * np.log10(np.mean(noisy[gap] ** 2))
    after = 10 * np.log10(np.mean(y[gap] ** 2) + 1e-12)
    assert after < before - 10, f"noise floor only dropped from {before:.1f} to {after:.1f} dB"
