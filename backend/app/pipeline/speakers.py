"""Speaker embeddings and the per-speaker consistency check.

Embeddings: pyannote/wespeaker-voxceleb-resnet34-LM (256-d, not gated) through
pyannote.audio, computed on in-memory 16 kHz audio. They are stored per clip
(float16) so a speaker's centroid can be rebuilt from every file they appear in.

Consistency: per speaker, a robust centroid (mean of L2-normalised embeddings,
re-estimated twice without the least similar 10%). A clip whose cosine
similarity to the centroid is below `min_similarity` - or, once the speaker has
enough clips, far below the speaker's own typical similarity (median - 4·MAD) -
is flagged `speaker_mismatch` (needs_review if still pending). Similarities are
recomputed for all of the speaker's clips, since the centroid moves as data grows.
"""
from __future__ import annotations

import logging

import numpy as np
from sqlalchemy import select

from .. import engines
from ..db import session_scope
from ..devices import has_module, resolve_device
from ..models import Clip

log = logging.getLogger(__name__)

SPEAKER_FLAGS = {"speaker_mismatch"}
EMB_DIM = 256


class SpeakerEmbedder:
    def __init__(self, model_id: str, device: str):
        import torch
        from pyannote.audio import Inference, Model

        self.torch = torch
        self.name = model_id
        model = Model.from_pretrained(model_id)
        self.inference = Inference(model, window="whole", device=torch.device(device))

    def embed(self, audio16k: np.ndarray) -> np.ndarray:
        x = np.ascontiguousarray(audio16k, dtype=np.float32)
        if len(x) < 8000:  # very short clips: pad to 0.5 s
            x = np.pad(x, (0, 8000 - len(x)))
        e = self.inference({"waveform": self.torch.from_numpy(x)[None, :], "sample_rate": 16000})
        e = np.asarray(e, dtype=np.float32).reshape(-1)
        return e / (np.linalg.norm(e) + 1e-9)


_embedder: SpeakerEmbedder | None = None
_embedder_failed: str | None = None


def get_embedder(settings) -> SpeakerEmbedder | None:
    global _embedder, _embedder_failed
    if _embedder is None and _embedder_failed is None:
        if not has_module("pyannote.audio"):
            _embedder_failed = "pyannote.audio not installed"
        else:
            try:
                _embedder = SpeakerEmbedder(settings.speaker_check.model, resolve_device(settings.device))
                engines.set_status("speaker_embedding", settings.speaker_check.model)
            except Exception as exc:  # noqa: BLE001 - optional component
                _embedder_failed = f"{type(exc).__name__}: {exc}"
        if _embedder_failed:
            log.warning("speaker embeddings unavailable: %s", _embedder_failed)
            engines.set_status("speaker_embedding", f"unavailable: {_embedder_failed[:120]}")
    return _embedder


def to_blob(e: np.ndarray) -> bytes:
    return e.astype(np.float16).tobytes()


def from_blob(b: bytes) -> np.ndarray:
    e = np.frombuffer(b, dtype=np.float16).astype(np.float32)
    return e / (np.linalg.norm(e) + 1e-9)


# ---------------------------------------------------------------- pure maths (unit-tested)
def robust_centroid(embs: np.ndarray, iterations: int = 2, trim: float = 0.1) -> np.ndarray:
    c = embs.mean(axis=0)
    c /= np.linalg.norm(c) + 1e-9
    for _ in range(iterations):
        if len(embs) < 10:
            break
        sims = embs @ c
        keep = sims >= np.quantile(sims, trim)
        c = embs[keep].mean(axis=0)
        c /= np.linalg.norm(c) + 1e-9
    return c


def mismatch_mask(sims: np.ndarray, min_similarity: float, relative_from: int = 20) -> np.ndarray:
    mask = sims < min_similarity
    if len(sims) >= relative_from:
        med = float(np.median(sims))
        mad = float(np.median(np.abs(sims - med))) * 1.4826
        mask |= sims < med - 4 * max(mad, 0.02)
    return mask


# ---------------------------------------------------------------- DB-backed check
def check_speaker(speaker_id: int, settings) -> dict:
    """Recompute similarities + speaker_mismatch flags for every clip of one speaker."""
    cfg = settings.speaker_check
    with session_scope() as s:
        rows = s.execute(select(Clip.id, Clip.status, Clip.speaker_embedding)
                         .where(Clip.speaker_id == speaker_id, Clip.speaker_embedding.is_not(None))).all()
    if len(rows) < cfg.min_clips:
        return {"speaker_id": speaker_id, "clips": len(rows), "flagged": 0, "skipped": "too few clips"}
    ids = [r[0] for r in rows]
    embs = np.stack([from_blob(r[2]) for r in rows])
    basis = np.array([r[1] != "rejected" for r in rows])  # rejected clips don't shape the centroid
    centroid = robust_centroid(embs[basis] if basis.sum() >= cfg.min_clips else embs)
    sims = embs @ centroid
    mask = mismatch_mask(sims, cfg.min_similarity)
    flagged = 0
    with session_scope(write=True) as s:
        for clip_id, sim, bad in zip(ids, sims, mask):
            clip = s.get(Clip, clip_id)
            clip.speaker_similarity = round(float(sim), 4)
            flags = set(clip.flags()) - SPEAKER_FLAGS
            if bad:
                flags.add("speaker_mismatch")
                flagged += 1
            clip.set_flags(flags)
            if clip.status == "pending":
                clip.needs_review = bool(flags)
    return {"speaker_id": speaker_id, "clips": len(rows), "flagged": flagged,
            "median_similarity": round(float(np.median(sims)), 3)}
