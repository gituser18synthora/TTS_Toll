"""Clip quality: DNSMOS P.835 (SIG / BAK / OVRL) with an SNR fallback, plus an optional
AudioSet tagger that flags music and background babble.

DNSMOS: Microsoft's sig_bak_ovr.onnx (DNS-Challenge), downloaded once into
models/dnsmos/. Scored on 9.01 s windows (1 s hop, short clips are tiled, as in
the reference implementation) and mapped with the official polynomial.
quality_score = OVRL on a 1-5 scale.

Fallback when DNSMOS is unavailable: an energy-based SNR estimate mapped onto the
same 1-5 scale (5 dB -> 1.0, 40 dB -> 5.0) so the thresholds keep working.

Tagger (MIT AST fine-tuned on AudioSet, via transformers): music / singing /
instrument and hubbub / chatter / crowd probabilities. ~2 s per clip on CPU,
so by default it only runs on a GPU (quality.tagger.enabled: auto).
"""
from __future__ import annotations

import logging
import os
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .. import engines
from ..devices import has_module, resolve_device

log = logging.getLogger(__name__)

SR = 16000
DNSMOS_URL = "https://github.com/microsoft/DNS-Challenge/raw/master/DNSMOS/DNSMOS/sig_bak_ovr.onnx"
WINDOW = int(9.01 * SR)
_P_OVR = np.poly1d([-0.06766283, 1.11546468, 0.04602535])
_P_SIG = np.poly1d([-0.08397278, 1.22083953, 0.0052439])
_P_BAK = np.poly1d([-0.13166888, 1.60915514, -0.39604546])

MUSIC_LABELS = ("Music", "Musical instrument", "Singing", "Song", "Background music")
BABBLE_LABELS = ("Hubbub, speech noise, speech babble", "Chatter", "Crowd")


@dataclass
class QualityResult:
    score: float                     # 1-5 (DNSMOS OVRL or SNR-mapped)
    method: str                      # dnsmos | snr
    details: dict = field(default_factory=dict)


def dnsmos_path(models_dir: Path) -> Path:
    return models_dir / "dnsmos" / "sig_bak_ovr.onnx"


def download_dnsmos(models_dir: Path) -> Path:
    path = dnsmos_path(models_dir)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".part")
        log.info("downloading DNSMOS model -> %s", path)
        urllib.request.urlretrieve(DNSMOS_URL, tmp)
        os.replace(tmp, path)
    return path


class Dnsmos:
    def __init__(self, models_dir: Path):
        import onnxruntime as ort

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = min(4, os.cpu_count() or 1)
        providers = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in ort.get_available_providers()]
        self.session = ort.InferenceSession(str(download_dnsmos(models_dir)), sess_options=opts, providers=providers)

    def score(self, audio16k: np.ndarray) -> QualityResult:
        x = audio16k.astype(np.float32)
        if len(x) == 0:
            return QualityResult(1.0, "dnsmos", {"sig": 1.0, "bak": 1.0, "ovrl": 1.0})
        while len(x) < WINDOW:  # reference implementation tiles short clips
            x = np.concatenate([x, x])
        hops = int(np.floor(len(x) / SR) - 9.01) + 1
        batch = np.stack([x[i * SR:i * SR + WINDOW] for i in range(hops)])
        raw = self.session.run(None, {"input_1": batch})[0]  # (hops, 3): sig, bak, ovr
        sig, bak, ovr = raw.mean(axis=0)
        details = {"sig": round(float(_P_SIG(sig)), 3), "bak": round(float(_P_BAK(bak)), 3),
                   "ovrl": round(float(_P_OVR(ovr)), 3)}
        return QualityResult(details["ovrl"], "dnsmos", details)


def snr_quality(audio: np.ndarray, sr: int) -> QualityResult:
    """Energy-percentile SNR estimate mapped onto 1-5 (no model needed)."""
    frame = max(1, int(0.02 * sr))
    n = len(audio) // frame
    if n < 5:
        return QualityResult(1.0, "snr", {"snr_db": 0.0})
    energy = np.mean(np.square(audio[:n * frame].reshape(n, frame), dtype=np.float64), axis=1)
    db = 10 * np.log10(np.maximum(energy, 1e-12))
    noise, speech = np.percentile(db, 10), np.percentile(db, 90)
    snr = float(speech - noise)
    return QualityResult(round(float(np.clip(1 + 4 * (snr - 5) / 35, 1, 5)), 3), "snr", {"snr_db": round(snr, 1)})


class AudioTagger:
    def __init__(self, model_id: str, device: str, cache_dir: str | None = None):
        import torch
        from transformers import ASTFeatureExtractor, ASTForAudioClassification

        self.torch = torch
        self.device = device
        self.extractor = ASTFeatureExtractor.from_pretrained(model_id, cache_dir=cache_dir)
        self.model = ASTForAudioClassification.from_pretrained(model_id, cache_dir=cache_dir).to(device).eval()
        names = self.model.config.id2label
        self.music_ids = [i for i, n in names.items() if n in MUSIC_LABELS]
        self.babble_ids = [i for i, n in names.items() if n in BABBLE_LABELS]
        self.window = int(10.24 * SR)

    def tag(self, audio16k: np.ndarray) -> dict:
        x = audio16k.astype(np.float32)
        windows = [x[:self.window]] if len(x) <= self.window else [x[:self.window], x[-self.window:]]
        inputs = self.extractor(windows, sampling_rate=SR, return_tensors="pt").to(self.device)
        with self.torch.inference_mode():
            probs = self.torch.sigmoid(self.model(**inputs).logits).max(dim=0).values.cpu().numpy()
        return {"music": round(float(probs[self.music_ids].max()), 3) if self.music_ids else 0.0,
                "babble": round(float(probs[self.babble_ids].max()), 3) if self.babble_ids else 0.0}


# ---------------------------------------------------------------- cached factories
_dnsmos: Dnsmos | None = None
_dnsmos_failed: str | None = None
_tagger: AudioTagger | None = None
_tagger_skip: str | None = None


def get_dnsmos(settings) -> Dnsmos | None:
    global _dnsmos, _dnsmos_failed
    if settings.quality.engine != "dnsmos":
        engines.set_status("quality", "snr (configured)")
        return None
    if _dnsmos is None and _dnsmos_failed is None:
        try:
            _dnsmos = Dnsmos(settings.models_dir)
            engines.set_status("quality", "dnsmos")
        except Exception as exc:  # noqa: BLE001 - offline / missing onnxruntime -> SNR fallback
            _dnsmos_failed = f"{type(exc).__name__}: {exc}"
            log.warning("DNSMOS unavailable (%s); using SNR estimate", _dnsmos_failed)
            engines.set_status("quality", f"snr (DNSMOS unavailable: {_dnsmos_failed[:120]})")
    return _dnsmos


def tagger_wanted(settings) -> tuple[bool, str]:
    enabled = settings.quality.tagger.enabled
    if enabled is False:
        return False, "disabled in config"
    if enabled == "auto" and resolve_device(settings.device) != "cuda":
        return False, "skipped on CPU (quality.tagger.enabled: auto)"
    if not has_module("transformers"):
        return False, "transformers not installed"
    return True, ""


def get_tagger(settings) -> AudioTagger | None:
    global _tagger, _tagger_skip
    wanted, why = tagger_wanted(settings)
    if not wanted:
        engines.set_status("tagger", why)
        return None
    if _tagger is None and _tagger_skip is None:
        try:
            _tagger = AudioTagger(settings.quality.tagger.model, resolve_device(settings.device))
            engines.set_status("tagger", f"ast ({settings.quality.tagger.model})")
        except Exception as exc:  # noqa: BLE001
            _tagger_skip = f"{type(exc).__name__}: {exc}"
            log.warning("audio tagger unavailable: %s", _tagger_skip)
            engines.set_status("tagger", f"unavailable: {_tagger_skip[:120]}")
    return _tagger
