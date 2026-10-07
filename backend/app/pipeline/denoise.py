"""Speech enhancement with DeepFilterNet3 (48 kHz).

The original upload is decoded to 48 kHz mono (full bandwidth, not the 24 kHz
copy), enhanced chunk by chunk (each chunk gets 1 s of context on both sides,
neighbouring chunks are cross-faded), and resampled to 24 kHz PCM16 as
data/enhanced/src_XXXXX.wav. The plain converted file is kept for A/B listening.

DeepFilterNet 0.5.6 still imports `torchaudio.backend.common`, removed from
torchaudio >= 2.1; it is only used for a type hint, so a stand-in module is
registered before importing it. Weights live in models/deepfilternet/.
If DeepFilterNet is missing or fails to load, the step is skipped with a warning
and clips are cut from the converted audio.
"""
from __future__ import annotations

import io
import logging
import os
import sys
import types
import urllib.request
import zipfile
from collections.abc import Callable
from pathlib import Path

import numpy as np
import soundfile as sf

from .. import engines

log = logging.getLogger(__name__)

DF_SR = 48000
MODEL_NAME = "DeepFilterNet3"
MODEL_URL = f"https://github.com/Rikorose/DeepFilterNet/raw/main/models/{MODEL_NAME}.zip"


def _install_torchaudio_shim() -> None:
    try:
        import torchaudio.backend.common  # noqa: F401
        return
    except ImportError:
        pass
    backend = types.ModuleType("torchaudio.backend")
    common = types.ModuleType("torchaudio.backend.common")

    class AudioMetaData:  # noqa: D401 - placeholder for a type hint in df.io
        pass

    common.AudioMetaData = AudioMetaData
    backend.common = common
    sys.modules.setdefault("torchaudio.backend", backend)
    sys.modules.setdefault("torchaudio.backend.common", common)


def model_dir(models_dir: Path) -> Path:
    return models_dir / "deepfilternet" / MODEL_NAME


def download_model(models_dir: Path) -> Path:
    target = model_dir(models_dir)
    if (target / "config.ini").exists():
        return target
    log.info("downloading %s -> %s", MODEL_NAME, target.parent)
    with urllib.request.urlopen(MODEL_URL, timeout=120) as resp:
        data = resp.read()
    target.parent.mkdir(parents=True, exist_ok=True)
    zipfile.ZipFile(io.BytesIO(data)).extractall(target.parent)
    return target


class DeepFilter:
    def __init__(self, models_dir: Path):
        _install_torchaudio_shim()
        import df.logger
        import torch
        from df.enhance import enhance, init_df

        df.logger.get_commit_hash = lambda: None  # avoid noisy `git rev-parse` subprocesses
        df.logger.get_branch_name = lambda: None
        self.torch = torch
        self._enhance = enhance
        self.model, self.state, _ = init_df(model_base_dir=str(download_model(models_dir)), log_file=None,
                                            log_level="WARNING", post_filter=False)

    def enhance(self, audio48k: np.ndarray, atten_lim_db: float | None) -> np.ndarray:
        x = self.torch.from_numpy(np.ascontiguousarray(audio48k, dtype=np.float32))[None, :]
        y = self._enhance(self.model, self.state, x, atten_lim_db=atten_lim_db)
        return y[0].numpy().astype(np.float32)


def process_in_chunks(reader: Callable[[int, int], np.ndarray], total: int, writer: Callable[[np.ndarray], None],
                      fn: Callable[[np.ndarray], np.ndarray], *, chunk: int, context: int, fade: int,
                      on_progress: Callable[[float], None] | None = None,
                      check_cancel: Callable[[], None] | None = None) -> None:
    """Run `fn` over [0, total) in chunks with context, cross-fading 2*fade samples at every seam.

    `fn` must return an array of the same length as its input. With an identity `fn` the
    output equals the input exactly (see tests).
    """
    pending: np.ndarray | None = None
    for start in range(0, total, chunk):
        end = min(total, start + chunk)
        a, b = max(0, start - context), min(total, end + context)
        y = fn(reader(a, b))
        lo, hi = max(0, start - fade), min(total, end + fade)
        piece = y[lo - a:hi - a].copy()
        if pending is not None:
            n = min(len(pending), len(piece))
            ramp = np.linspace(0.0, 1.0, n, dtype=np.float32)
            piece[:n] = pending[:n] * (1 - ramp) + piece[:n] * ramp
        if hi < total:
            writer(piece[:-2 * fade])
            pending = piece[-2 * fade:]
        else:
            writer(piece)
            pending = None
        if on_progress:
            on_progress(end / total)
        if check_cancel:
            check_cancel()


_df: DeepFilter | None = None
_df_failed: str | None = None


def get_denoiser(settings) -> DeepFilter | None:
    global _df, _df_failed
    if settings.denoise.engine == "none":
        engines.set_status("denoise", "disabled in config")
        return None
    if _df is None and _df_failed is None:
        try:
            _df = DeepFilter(settings.models_dir)
            engines.set_status("denoise", MODEL_NAME)
        except Exception as exc:  # noqa: BLE001 - optional component: skip with a warning
            _df_failed = f"{type(exc).__name__}: {exc}"
            log.warning("DeepFilterNet unavailable (%s); denoising skipped", _df_failed)
            engines.set_status("denoise", f"skipped (DeepFilterNet unavailable: {_df_failed[:120]})")
    return _df


def enhance_file(denoiser: DeepFilter, wav48: Path, out_wav48: Path, *, chunk_sec: float,
                 atten_lim_db: float | None, on_progress=None, check_cancel=None) -> None:
    """Enhance a 48 kHz mono WAV into a float32 48 kHz WAV (streaming, bounded memory)."""
    info = sf.info(str(wav48))
    if info.samplerate != DF_SR:
        raise ValueError(f"DeepFilterNet input must be {DF_SR} Hz")
    tmp = out_wav48.with_name(out_wav48.stem + ".part.wav")
    with sf.SoundFile(str(wav48)) as src, sf.SoundFile(str(tmp), "w", DF_SR, 1, "FLOAT") as dst:
        def reader(a: int, b: int) -> np.ndarray:
            src.seek(a)
            x = src.read(b - a, dtype="float32", always_2d=True)
            return x.mean(axis=1) if x.shape[1] > 1 else x[:, 0]

        process_in_chunks(reader, info.frames, dst.write, lambda x: denoiser.enhance(x, atten_lim_db),
                          chunk=int(chunk_sec * DF_SR), context=DF_SR, fade=int(0.05 * DF_SR),
                          on_progress=on_progress, check_cancel=check_cancel)
    os.replace(tmp, out_wav48)
