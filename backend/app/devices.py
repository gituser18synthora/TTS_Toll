"""Device resolution and optional-dependency probing (all imports are lazy)."""
from __future__ import annotations

import importlib.util
import logging
import shutil
import subprocess
from functools import lru_cache

log = logging.getLogger(__name__)


def has_module(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


@lru_cache(maxsize=1)
def torch_cuda_available() -> bool:
    if not has_module("torch"):
        return False
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:  # pragma: no cover - broken CUDA installs
        return False


@lru_cache(maxsize=1)
def ctranslate2_cuda_count() -> int:
    if not has_module("ctranslate2"):
        return 0
    try:
        if torch_cuda_available():
            import torch  # noqa: F401  (loads the cuBLAS/cuDNN libs ctranslate2 needs)
        import ctranslate2

        return int(ctranslate2.get_cuda_device_count())
    except Exception:  # pragma: no cover
        return 0


def resolve_device(preference: str) -> str:
    """auto|cpu|cuda -> cpu|cuda. Falls back to cpu (with a warning) if CUDA is missing."""
    if preference == "cpu":
        return "cpu"
    cuda = torch_cuda_available() or ctranslate2_cuda_count() > 0
    if preference == "cuda" and not cuda:
        log.warning("device: cuda requested but no CUDA device is available; using cpu")
    return "cuda" if cuda else "cpu"


@lru_cache(maxsize=4)
def ffmpeg_ok(ffmpeg: str = "ffmpeg") -> bool:
    return shutil.which(ffmpeg) is not None


@lru_cache(maxsize=4)
def ffmpeg_has_soxr(ffmpeg: str = "ffmpeg") -> bool:
    try:
        out = subprocess.run([ffmpeg, "-hide_banner", "-buildconf"], capture_output=True, text=True, timeout=20)
        return "--enable-libsoxr" in out.stdout
    except Exception:
        return False


def capability_report(settings) -> dict:
    """What is installed — shown in the UI and by `python -m app.cli doctor`."""
    return {
        "ffmpeg": ffmpeg_ok(settings.audio.ffmpeg),
        "ffmpeg_soxr": ffmpeg_has_soxr(settings.audio.ffmpeg),
        "torch": has_module("torch"),
        "silero_vad": has_module("silero_vad") and has_module("onnxruntime"),
        "faster_whisper": has_module("faster_whisper"),
        "pyloudnorm": has_module("pyloudnorm"),
        "deepfilternet": has_module("df") and (settings.models_dir / "deepfilternet" / "DeepFilterNet3").exists(),
        "dnsmos": (settings.models_dir / "dnsmos" / "sig_bak_ovr.onnx").exists(),
        "transformers": has_module("transformers"),
        "datasets": has_module("datasets"),
    }
