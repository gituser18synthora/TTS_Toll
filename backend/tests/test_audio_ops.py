"""Per-clip audio processing: trimming, loudness, limiter, WAV format."""
from __future__ import annotations

import numpy as np
import pytest
import soundfile as sf

from app.pipeline.audio_ops import limit_peaks, measure_lufs, process_clip, trim_bounds, write_wav

SR = 24000


def tone_burst(seconds: float, amp: float = 0.3, freq: float = 220.0) -> np.ndarray:
    t = np.arange(int(seconds * SR)) / SR
    return (amp * np.sin(2 * np.pi * freq * t) * (0.6 + 0.4 * np.sin(2 * np.pi * 3 * t))).astype(np.float32)


def test_trim_keeps_requested_padding():
    x = np.concatenate([np.zeros(SR), tone_burst(2.0), np.zeros(SR)]).astype(np.float32)
    a, b = trim_bounds(x, SR, top_db=40, keep_sec=0.15)
    assert abs(a / SR - (1.0 - 0.15)) < 0.03
    assert abs(b / SR - (3.0 + 0.15)) < 0.03


def test_trim_never_cuts_into_signal():
    x = tone_burst(3.0)
    a, b = trim_bounds(x, SR, keep_sec=0.15)
    assert a == 0 and b == len(x)


@pytest.mark.parametrize("amp", [0.01, 0.1, 0.5])
def test_loudness_normalized_to_target(amp):
    res = process_clip(tone_burst(4.0, amp=amp), SR, keep_sec=0.15, top_db=40, target_lufs=-23.0, peak_dbfs=-1.0, max_gain_db=40)
    assert measure_lufs(res.audio, SR) == pytest.approx(-23.0, abs=0.2)
    assert np.abs(res.audio).max() <= 10 ** (-1 / 20) + 1e-6


def test_limiter_guarantees_ceiling():
    rng = np.random.default_rng(0)
    x = (rng.standard_normal(SR * 3) * 0.4).astype(np.float32)
    x[SR:SR + 50] = 1.8  # big transient
    y, limited = limit_peaks(x, SR, ceiling_db=-1.0)
    assert limited
    assert np.abs(y).max() <= 10 ** (-1 / 20) + 1e-6
    # gain only reduced around peaks: quiet parts far from any peak stay untouched
    quiet = np.abs(x) < 0.05
    far = np.ones_like(quiet)
    far[: SR + SR // 5] = False
    assert np.allclose(y[quiet & far][:1000], x[quiet & far][:1000], atol=0.05)


def test_limiter_noop_below_ceiling():
    x = tone_burst(1.0, amp=0.2)
    y, limited = limit_peaks(x, SR, ceiling_db=-1.0)
    assert not limited and np.array_equal(x, y)


def test_write_wav_is_24k_mono_pcm16(tmp_path):
    path = tmp_path / "c.wav"
    write_wav(path, tone_burst(1.0), SR)
    info = sf.info(str(path))
    assert (info.samplerate, info.channels, info.subtype) == (SR, 1, "PCM_16")
    assert not list(tmp_path.glob("*.part.wav"))
