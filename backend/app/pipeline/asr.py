"""ASR backends. Phase 1: faster-whisper (primary). Models load lazily and are cached per process."""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass

import numpy as np

from .. import engines
from ..devices import has_module, resolve_device

log = logging.getLogger(__name__)

ASR_SR = 16000


@dataclass
class AsrResult:
    text: str
    avg_logprob: float | None = None
    no_speech_prob: float | None = None
    compression_ratio: float | None = None


class AsrUnavailable(RuntimeError):
    pass


class WhisperASR:
    def __init__(self, model_name: str, device: str, compute_type: str, download_root: str,
                 beam_size: int = 5, cpu_threads: int = 0):
        if not has_module("faster_whisper"):
            raise AsrUnavailable("faster-whisper is not installed")
        if device == "cuda":
            import torch  # noqa: F401  (makes torch's bundled cuBLAS/cuDNN visible to ctranslate2)
        from faster_whisper import WhisperModel
        from faster_whisper.utils import download_model

        self.name = f"faster-whisper:{model_name}"
        self.beam_size = beam_size
        try:  # weights already in models/ -> load them without touching the network (offline servers)
            source = download_model(model_name, local_files_only=True, cache_dir=download_root)
        except Exception:  # noqa: BLE001 - not cached yet (or a local path): let WhisperModel resolve it
            source = model_name
        log.info("loading %s on %s (%s)", self.name, device, compute_type)
        self.model = WhisperModel(source, device=device, compute_type=compute_type,
                                  download_root=download_root, cpu_threads=cpu_threads)

    def transcribe(self, audio16k: np.ndarray, language: str | None,
                   initial_prompt: str | None = None) -> AsrResult:
        segments, _info = self.model.transcribe(
            audio16k.astype(np.float32),
            language=language,
            task="transcribe",
            beam_size=self.beam_size,
            vad_filter=False,                  # clips are already VAD-trimmed
            condition_on_previous_text=False,  # clips are independent
            without_timestamps=True,
            initial_prompt=initial_prompt,
        )
        segs = list(segments)
        text = " ".join(s.text.strip() for s in segs).strip()
        if not segs:
            return AsrResult(text="")
        # duration-weighted average log-prob; worst no-speech / compression across segments
        weights = np.array([max(s.end - s.start, 0.01) for s in segs])
        avg_lp = float(np.average([s.avg_logprob for s in segs], weights=weights))
        return AsrResult(
            text=text,
            avg_logprob=avg_lp,
            no_speech_prob=float(max(s.no_speech_prob for s in segs)),
            compression_ratio=float(max(s.compression_ratio for s in segs)),
        )


class IndicConformerASR:
    """AI4Bharat IndicConformer 600M multilingual (hybrid CTC/RNNT, ONNX inside a
    trust_remote_code transformers model). Gated on HuggingFace: needs HF_TOKEN
    (from .env) for an account that accepted the model's terms."""

    def __init__(self, model_id: str, decoding: str):
        if not os.environ.get("HF_TOKEN"):
            raise AsrUnavailable("HF_TOKEN not set (add it to .env)")
        if not has_module("transformers"):
            raise AsrUnavailable("transformers is not installed")
        import torch
        from transformers import AutoModel

        self.torch = torch
        self.decoding = decoding
        self.name = f"indicconformer:{decoding}"
        log.info("loading %s (%s)", model_id, decoding)
        self.model = AutoModel.from_pretrained(model_id, trust_remote_code=True)

    def transcribe(self, audio16k: np.ndarray, language: str | None, initial_prompt: str | None = None) -> AsrResult:
        wav = self.torch.from_numpy(np.ascontiguousarray(audio16k, dtype=np.float32)).unsqueeze(0)
        with self.torch.inference_mode():
            out = self.model(wav, language, self.decoding)
        if isinstance(out, (list, tuple)):
            out = out[0] if out else ""
        return AsrResult(text=str(out or "").strip())


_cache: dict[tuple, object] = {}
_failed: dict[tuple, str] = {}


def _whisper(settings, model: str) -> WhisperASR | None:
    cfg = settings.asr.primary
    device = resolve_device(settings.device)
    compute = cfg.compute_type_gpu if device == "cuda" else cfg.compute_type_cpu
    key = ("whisper", model, device, compute, cfg.beam_size)
    if key not in _cache and key not in _failed:
        try:
            _cache[key] = WhisperASR(model, device, compute, str(settings.models_dir / "whisper"),
                                     beam_size=cfg.beam_size, cpu_threads=cfg.cpu_threads)
        except Exception as exc:  # noqa: BLE001 - missing package, download failure, bad model name
            _failed[key] = f"{type(exc).__name__}: {exc}"
            log.warning("whisper %s unavailable: %s", model, _failed[key])
    return _cache.get(key)  # type: ignore[return-value]


def get_primary_asr(settings) -> WhisperASR | None:
    """Return the configured primary ASR, or None when ASR is disabled/unavailable."""
    cfg = settings.asr.primary
    if cfg.engine == "none":
        engines.set_status("asr_primary", "disabled in config")
        return None
    device = resolve_device(settings.device)
    engine = _whisper(settings, cfg.model_gpu if device == "cuda" else cfg.model_cpu)
    engines.set_status("asr_primary", engine.name if engine else "unavailable")
    return engine


def _indicconformer(settings) -> tuple[IndicConformerASR | None, str]:
    cfg = settings.asr.secondary.indicconformer
    key = ("indicconformer", cfg.model, cfg.decoding)
    if key not in _cache and key not in _failed:
        try:
            _cache[key] = IndicConformerASR(cfg.model, cfg.decoding)
        except Exception as exc:  # noqa: BLE001 - gated / no token / incompatible remote code
            reason = str(exc) if isinstance(exc, AsrUnavailable) else f"{type(exc).__name__}: {exc}"
            if "gated" in reason.lower() or "403" in reason or "401" in reason:
                reason = f"no access to {cfg.model}; accept its terms on huggingface.co with the HF_TOKEN account"
            _failed[key] = reason[:300]
            log.warning("IndicConformer unavailable (%s); using the whisper fallback", _failed[key])
    return _cache.get(key), _failed.get(key, "")  # type: ignore[return-value]


def get_secondary_asr(settings, language: str):
    """Second opinion for the agreement score: IndicConformer for its languages, else a
    second whisper pass with a different model size. None if disabled/unavailable."""
    cfg = settings.asr.secondary
    if cfg.engine == "none":
        engines.set_status("asr_secondary", "disabled in config")
        return None
    note = ""
    if cfg.engine == "indicconformer" and language in cfg.indicconformer.languages:
        engine, why = _indicconformer(settings)
        if engine is not None:
            engines.set_status("asr_secondary", f"{engine.name} for {','.join(cfg.indicconformer.languages)}")
            return engine
        note = f" (IndicConformer: {why})"
    device = resolve_device(settings.device)
    size = cfg.whisper_gpu if device == "cuda" else cfg.whisper_cpu
    primary = settings.asr.primary.model_gpu if device == "cuda" else settings.asr.primary.model_cpu
    if size == primary:
        log.warning("secondary whisper size equals the primary (%s): agreement scores will be optimistic", size)
    engine = _whisper(settings, size)
    engines.set_status("asr_secondary", (engine.name if engine else "unavailable") + note)
    return engine


def secondary_fallback_reason(settings) -> str:
    """Why IndicConformer is not used (empty if it is, or was never tried)."""
    cfg = settings.asr.secondary.indicconformer
    return _failed.get(("indicconformer", cfg.model, cfg.decoding), "")


def review_flags(result: AsrResult, duration: float, cfg) -> list[str]:
    """Heuristics that mark a transcript as worth a human look."""
    flags: list[str] = []
    if not result.text:
        return ["empty_transcript"]
    if result.avg_logprob is not None and result.avg_logprob < cfg.min_avg_logprob:
        flags.append("low_asr_confidence")
    if result.no_speech_prob is not None and result.no_speech_prob > cfg.max_no_speech_prob:
        flags.append("asr_no_speech")
    if result.compression_ratio is not None and result.compression_ratio > cfg.max_compression_ratio:
        flags.append("asr_repetition")
    chars = len(result.text.replace(" ", ""))
    if duration >= 3.0 and chars / duration < cfg.min_chars_per_sec:
        flags.append("sparse_transcript")
    return flags
