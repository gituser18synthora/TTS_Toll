"""Per-clip audio processing: trim, loudness-normalize, peak-limit, write 16-bit WAV."""
from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.ndimage import minimum_filter1d, uniform_filter1d


def read_segment(path: Path, start: float, end: float) -> tuple[np.ndarray, int]:
    """Read [start, end) seconds as mono float32 without loading the whole file."""
    info = sf.info(str(path))
    sr = info.samplerate
    a = max(0, int(round(start * sr)))
    b = min(info.frames, int(round(end * sr)))
    x, sr = sf.read(str(path), start=a, stop=max(a, b), dtype="float32", always_2d=True)
    return x.mean(axis=1) if x.shape[1] > 1 else x[:, 0], sr


def read_wav(path: Path) -> tuple[np.ndarray, int]:
    x, sr = sf.read(str(path), dtype="float32", always_2d=True)
    return (x.mean(axis=1) if x.shape[1] > 1 else x[:, 0]), sr


def frame_rms_db(x: np.ndarray, sr: int, frame_ms: float = 25.0, hop_ms: float = 10.0) -> tuple[np.ndarray, int, int]:
    frame = max(1, int(sr * frame_ms / 1000))
    hop = max(1, int(sr * hop_ms / 1000))
    if len(x) < frame:
        rms = np.sqrt(np.mean(np.square(x, dtype=np.float64))) if len(x) else 0.0
        return np.array([20 * np.log10(max(rms, 1e-10))]), frame, hop
    n = 1 + (len(x) - frame) // hop
    idx = np.arange(frame)[None, :] + hop * np.arange(n)[:, None]
    rms = np.sqrt(np.mean(np.square(x[idx], dtype=np.float64), axis=1))
    return 20 * np.log10(np.maximum(rms, 1e-10)), frame, hop


def trim_bounds(x: np.ndarray, sr: int, top_db: float = 40.0, keep_sec: float = 0.15,
                floor_db: float = -65.0) -> tuple[int, int]:
    """Sample bounds that drop leading/trailing silence but keep `keep_sec` around the speech."""
    if len(x) == 0:
        return 0, 0
    db, frame, hop = frame_rms_db(x, sr)
    threshold = max(float(db.max()) - top_db, floor_db)
    active = np.flatnonzero(db >= threshold)
    if active.size == 0:
        return 0, len(x)
    keep = int(keep_sec * sr)
    first = int(active[0]) * hop
    last = min(len(x), int(active[-1]) * hop + frame)
    return max(0, first - keep), min(len(x), last + keep)


def measure_lufs(x: np.ndarray, sr: int) -> float | None:
    """Integrated loudness (ITU-R BS.1770). None if too short or silent."""
    import pyloudnorm

    if len(x) < int(0.4 * sr) + 1:
        return None
    try:
        lufs = float(pyloudnorm.Meter(sr).integrated_loudness(x.astype(np.float64)))
    except ValueError:
        return None
    return lufs if np.isfinite(lufs) else None


def limit_peaks(x: np.ndarray, sr: int, ceiling_db: float = -1.0, lookahead_ms: float = 5.0,
                hold_ms: float = 50.0) -> tuple[np.ndarray, bool]:
    """Look-ahead peak limiter. Guarantees |y| <= ceiling.

    Gain needed per sample g = min(1, ceiling/|x|) is turned into a smooth
    envelope: a sliding minimum over [n - hold, n + lookahead] followed by a
    moving average over +/- lookahead. Because every averaged value is <= g[n]
    (hold >= lookahead), the ceiling holds exactly, with ~10 ms gain ramps.
    """
    ceiling = 10 ** (ceiling_db / 20)
    peak = float(np.max(np.abs(x))) if len(x) else 0.0
    if peak <= ceiling:
        return x, False
    g = np.minimum(1.0, ceiling / np.maximum(np.abs(x), 1e-12))
    look = max(1, int(sr * lookahead_ms / 1000))
    hold = max(look, int(sr * hold_ms / 1000))
    if (hold - look) % 2:
        hold += 1
    k = (hold + look) // 2
    centered = minimum_filter1d(g, size=2 * k + 1, mode="nearest")
    shift = (hold - look) // 2
    windowed = np.empty_like(centered)  # windowed[n] = min g[n-hold .. n+look]
    windowed[shift:] = centered[: len(centered) - shift] if shift else centered
    windowed[:shift] = centered[0]
    gain = uniform_filter1d(windowed, size=2 * look + 1, mode="nearest")
    y = (x * gain).astype(np.float32)
    np.clip(y, -ceiling, ceiling, out=y)
    return y, True


@dataclass
class ClipResult:
    audio: np.ndarray
    trim_start: int          # samples removed at the start
    trim_end: int            # samples removed at the end
    lufs_in: float | None
    gain_db: float
    limited: bool


def process_clip(x: np.ndarray, sr: int, *, keep_sec: float, top_db: float, target_lufs: float,
                 peak_dbfs: float, max_gain_db: float) -> ClipResult:
    a, b = trim_bounds(x, sr, top_db=top_db, keep_sec=keep_sec)
    y = x[a:b].astype(np.float32)
    lufs = measure_lufs(y, sr)
    gain_db = 0.0
    if lufs is not None:
        gain_db = float(np.clip(target_lufs - lufs, -60.0, max_gain_db))
        y = (y * 10 ** (gain_db / 20)).astype(np.float32)
    y, limited = limit_peaks(y, sr, ceiling_db=peak_dbfs)
    return ClipResult(audio=y, trim_start=a, trim_end=len(x) - b, lufs_in=lufs, gain_db=gain_db, limited=limited)


def to_pcm16(x: np.ndarray) -> np.ndarray:
    return np.clip(np.round(x * 32767.0), -32768, 32767).astype(np.int16)


def write_wav(path: Path, x: np.ndarray, sr: int) -> None:
    """Atomic 16-bit PCM mono WAV write."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".part.wav")
    sf.write(str(tmp), to_pcm16(x), sr, subtype="PCM_16")
    os.replace(tmp, path)


def pcm_hash(x: np.ndarray) -> str:
    return hashlib.sha256(to_pcm16(x).tobytes()).hexdigest()


def resample_poly_to(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out:
        return x.astype(np.float32)
    from math import gcd

    from scipy.signal import resample_poly

    g = gcd(sr_in, sr_out)
    return resample_poly(x, sr_out // g, sr_in // g).astype(np.float32)
