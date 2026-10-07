"""Frame-level speech probabilities (32 ms frames at 16 kHz).

Primary engine: Silero VAD "sequence" ONNX model (runs ~500x real-time on CPU,
streams through the file block by block, so memory stays flat for long
recordings). Fallback: a simple adaptive energy detector, used automatically
if silero-vad / onnxruntime are not installed.
"""
from __future__ import annotations

import logging
from collections.abc import Callable, Iterator
from pathlib import Path

import numpy as np
import soundfile as sf

log = logging.getLogger(__name__)

VAD_SR = 16000
FRAME = 512                   # samples per frame at 16 kHz
FRAME_SEC = FRAME / VAD_SR    # 0.032 s
CONTEXT = 64                  # Silero prepends the last 64 samples of the previous frame
BLOCK_FRAMES = 512            # frames per ONNX call (~16 s)

_silero_model = None


def _load_silero():
    global _silero_model
    if _silero_model is None:
        import torch

        threads = torch.get_num_threads()
        from silero_vad import load_silero_vad  # NB: importing silero sets torch threads to 1

        torch.set_num_threads(threads)
        _silero_model = load_silero_vad(sequence=True)
    return _silero_model


def _blocks(path: Path, frames_per_block: int) -> Iterator[np.ndarray]:
    for block in sf.blocks(str(path), blocksize=frames_per_block * FRAME, dtype="float32", always_2d=True):
        yield block.mean(axis=1) if block.shape[1] > 1 else block[:, 0]


def _silero_probs(path: Path, total_frames: int, on_progress, check_cancel) -> np.ndarray:
    model = _load_silero()
    session = model.session
    h = np.zeros((1, 1, 128), dtype=np.float32)
    c = np.zeros((1, 1, 128), dtype=np.float32)
    context = np.zeros(CONTEXT, dtype=np.float32)
    out: list[np.ndarray] = []
    done = 0
    for audio in _blocks(path, BLOCK_FRAMES):
        n = (len(audio) + FRAME - 1) // FRAME
        frames = np.zeros((n, FRAME), dtype=np.float32)
        frames.reshape(-1)[: len(audio)] = audio
        contexts = np.empty((n, CONTEXT), dtype=np.float32)
        contexts[0] = context
        if n > 1:
            contexts[1:] = frames[:-1, -CONTEXT:]
        context = frames[-1, -CONTEXT:].copy()
        probs, h, c = session.run(["speech_probs", "hn", "cn"],
                                  {"input": np.concatenate((contexts, frames), axis=1), "h": h, "c": c})
        out.append(np.asarray(probs, dtype=np.float32).reshape(-1))
        done += n
        if on_progress:
            on_progress(min(1.0, done / max(total_frames, 1)))
        if check_cancel:
            check_cancel()
    return np.concatenate(out) if out else np.zeros(0, dtype=np.float32)


def _energy_probs(path: Path, total_frames: int, on_progress, check_cancel) -> np.ndarray:
    """Adaptive energy detector mapped to pseudo-probabilities via a sigmoid."""
    db_parts: list[np.ndarray] = []
    done = 0
    for audio in _blocks(path, 4096):
        n = (len(audio) + FRAME - 1) // FRAME
        frames = np.zeros((n, FRAME), dtype=np.float32)
        frames.reshape(-1)[: len(audio)] = audio
        rms = np.sqrt(np.mean(np.square(frames, dtype=np.float64), axis=1))
        db_parts.append(20 * np.log10(np.maximum(rms, 1e-10)))
        done += n
        if on_progress:
            on_progress(min(1.0, done / max(total_frames, 1)))
        if check_cancel:
            check_cancel()
    if not db_parts:
        return np.zeros(0, dtype=np.float32)
    db = np.concatenate(db_parts)
    floor, loud = np.percentile(db, 10), np.percentile(db, 95)
    thr = floor + max(6.0, 0.35 * (loud - floor))
    return (1.0 / (1.0 + np.exp(-(db - thr) / 2.0))).astype(np.float32)


def speech_probs(wav16k: Path, engine: str = "silero",
                 on_progress: Callable[[float], None] | None = None,
                 check_cancel: Callable[[], None] | None = None) -> tuple[np.ndarray, str]:
    """Return (per-frame speech probabilities, engine actually used)."""
    info = sf.info(str(wav16k))
    if info.samplerate != VAD_SR:
        raise ValueError(f"VAD input must be {VAD_SR} Hz, got {info.samplerate}")
    total_frames = (info.frames + FRAME - 1) // FRAME
    if engine == "silero":
        try:
            return _silero_probs(wav16k, total_frames, on_progress, check_cancel), "silero"
        except ImportError as exc:
            log.warning("Silero VAD unavailable (%s); falling back to energy VAD", exc)
    return _energy_probs(wav16k, total_frames, on_progress, check_cancel), "energy"
