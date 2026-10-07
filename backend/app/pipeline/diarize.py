"""Speaker diarization for files uploaded as "multiple speakers".

Primary: pyannote's speaker-diarization pipeline (pyannote/speaker-diarization-community-1,
gated: needs HF_TOKEN for an account that accepted its terms). It also reports overlapped
speech, which becomes the `overlap` flag / rejection.

Fallback (automatic when pyannote's pipeline is unavailable): sliding-window speaker
embeddings (wespeaker, not gated) on the VAD speech regions, agglomerative clustering
with a cosine-distance threshold (or a fixed number of speakers), frame-level smoothing.
Good for interviews / dialogues with distinct voices; no overlap detection.

Either way, speech that cannot be attributed with confidence (a lone blip between pauses,
a sub-0.3 s turn not surrounded by one speaker) gets no label and is kept out of all clips.

Output (work/src_XXXXX/diarization.json):
  turns    [[start, end, label], ...]   exclusive (one speaker at a time)
  overlaps [[start, end], ...]          where two or more speakers talk at once
  labels   {label: {seconds, turns, samples: [relative wav paths]}}
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass

import numpy as np

from .. import engines
from ..devices import has_module, resolve_device

log = logging.getLogger(__name__)

SR = 16000


@dataclass
class Diarization:
    turns: list[tuple[float, float, str]]
    overlaps: list[tuple[float, float]]
    engine: str


def merge_turns(turns: list[tuple[float, float, str]], min_turn: float = 0.3, join_gap: float = 0.3) -> list[tuple[float, float, str]]:
    """Sort and join same-speaker turns separated by <= join_gap. A turn still shorter than min_turn is
    a flicker: it joins the speaker whose turns touch it on both sides (gaps <= join_gap), otherwise it
    is dropped - speech of uncertain origin stays out of every clip (see segment.label_regions)."""
    out: list[list] = []
    for s, e, lab in sorted(turns):
        if out and out[-1][2] == lab and s - out[-1][1] <= join_gap:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e, lab])
    cleaned: list[list] = []
    for i, (s, e, lab) in enumerate(out):
        if e - s < min_turn:
            prev, nxt = (out[i - 1] if i else None), (out[i + 1] if i + 1 < len(out) else None)
            if not (prev and nxt and prev[2] == nxt[2] and s - prev[1] <= join_gap and nxt[0] - e <= join_gap):
                continue
            lab = prev[2]
        cleaned.append([s, e, lab])
    merged: list[tuple[float, float, str]] = []
    for s, e, lab in cleaned:
        if merged and merged[-1][2] == lab and s - merged[-1][1] <= join_gap:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e), lab)
        else:
            merged.append((s, e, lab))
    return merged


def relabel_by_duration(turns: list[tuple[float, float, str]]) -> list[tuple[float, float, str]]:
    """SPEAKER_00 = whoever talks most, SPEAKER_01 next, ..."""
    total: dict[str, float] = {}
    for s, e, lab in turns:
        total[lab] = total.get(lab, 0.0) + (e - s)
    order = {lab: f"SPEAKER_{i:02d}" for i, lab in enumerate(sorted(total, key=lambda k: -total[k]))}
    return [(s, e, order[lab]) for s, e, lab in turns]


class PyannoteDiarizer:
    def __init__(self, model_id: str, device: str):
        if not os.environ.get("HF_TOKEN"):
            raise RuntimeError("HF_TOKEN not set (add it to .env)")
        import torch
        from pyannote.audio import Pipeline

        self.torch = torch
        self.name = f"pyannote:{model_id}"
        self.pipeline = Pipeline.from_pretrained(model_id)
        if self.pipeline is None:  # pyannote returns None when access is denied
            raise RuntimeError(f"no access to {model_id}; accept its terms on huggingface.co with the HF_TOKEN account")
        self.pipeline.to(torch.device(device))

    def __call__(self, audio16k: np.ndarray, regions, num_speakers: int | None = None,
                 max_speakers: int | None = None) -> Diarization:
        wav = self.torch.from_numpy(np.ascontiguousarray(audio16k, dtype=np.float32))[None, :]
        kwargs = {"num_speakers": num_speakers} if num_speakers else {"max_speakers": max_speakers}
        out = self.pipeline({"waveform": wav, "sample_rate": SR}, **kwargs)
        full = getattr(out, "speaker_diarization", out)
        exclusive = getattr(out, "exclusive_speaker_diarization", None)
        turns = [(seg.start, seg.end, str(lab)) for seg, _, lab in (exclusive or full).itertracks(yield_label=True)]
        overlaps = [(seg.start, seg.end) for seg in full.get_overlap()] if hasattr(full, "get_overlap") else []
        return Diarization(relabel_by_duration(merge_turns(turns)), overlaps, self.name)


def absorb_minor_clusters(embs: np.ndarray, labels: np.ndarray, hop: float,
                          min_share: float = 0.05, min_seconds: float = 5.0) -> np.ndarray:
    """Short 1.5 s windows are noisy: a few of them form tiny clusters of their own. Clusters with less
    than `min_share` of the speech (or under `min_seconds`) join the most similar main cluster."""
    ids, counts = np.unique(labels, return_counts=True)
    total = counts.sum()
    major = [c for c, n in zip(ids, counts) if n / total >= min_share and n * hop >= min_seconds]
    if not major or len(major) == len(ids):
        return labels
    cent = {c: embs[labels == c].mean(axis=0) for c in ids}
    cent = {c: v / (np.linalg.norm(v) + 1e-9) for c, v in cent.items()}
    out = labels.copy()
    for c in ids:
        if c not in major:
            out[labels == c] = max(major, key=lambda m: float(cent[c] @ cent[m]))
    return out


class ClusteringDiarizer:
    """Fallback: embeddings of 1.5 s windows (0.75 s hop) inside speech, agglomerative clustering."""

    def __init__(self, embedder, threshold: float, attach_gap: float = 0.3):
        self.embedder = embedder
        self.threshold = threshold
        self.attach_gap = attach_gap
        self.name = "clustering (wespeaker embeddings)"

    def __call__(self, audio16k: np.ndarray, regions, num_speakers: int | None = None,
                 max_speakers: int | None = None) -> Diarization:
        from sklearn.cluster import AgglomerativeClustering

        win, hop = 1.5, 0.75
        windows: list[tuple[float, float]] = []
        owner: list[int] = []  # region index of each window
        for ri, (s, e) in enumerate(regions):
            if e - s < 0.5:
                continue
            t = s
            while True:
                windows.append((t, min(t + win, e)))
                owner.append(ri)
                if t + win >= e:
                    break
                t += hop
        if not windows:
            return Diarization([], [], self.name)
        embs = np.stack([self.embedder.embed(audio16k[int(a * SR):int(b * SR)]) for a, b in windows])
        if len(windows) == 1:
            labels = np.zeros(1, dtype=int)
        else:
            clusterer = AgglomerativeClustering(
                n_clusters=num_speakers if num_speakers else None,
                distance_threshold=None if num_speakers else self.threshold,
                metric="cosine", linkage="average")
            labels = clusterer.fit_predict(embs)
            if not num_speakers and max_speakers and len(set(labels)) > max_speakers:
                labels = AgglomerativeClustering(n_clusters=max_speakers, metric="cosine",
                                                 linkage="average").fit_predict(embs)
            if not num_speakers:
                labels = absorb_minor_clusters(embs, labels, hop)
        # each 0.1 s frame takes the label of the closest window centre of its own region
        centres = np.array([(a + b) / 2 for a, b in windows])
        owner_arr = np.array(owner)
        framed: dict[int, list[tuple[float, float, str]]] = {}
        for ri, (s, e) in enumerate(regions):
            mine = np.flatnonzero(owner_arr == ri)
            if not len(mine):
                continue
            framed[ri] = []
            t = s
            while t < e:
                f_end = min(t + 0.1, e)
                k = mine[int(np.argmin(np.abs(centres[mine] - (t + f_end) / 2)))]
                framed[ri].append((t, f_end, f"C{labels[k]}"))
                t = f_end
        turns = [t for ri in sorted(framed) for t in framed[ri]]
        # a region too short for a window takes the label of a labelled region it touches (gap <= attach_gap);
        # a blip between pauses - often the other speaker's back-channel "haan" - stays unattributed
        for ri, (s, e) in enumerate(regions):
            if ri in framed:
                continue
            near = set()
            if ri - 1 in framed and s - regions[ri - 1][1] <= self.attach_gap:
                near.add(framed[ri - 1][-1][2])
            if ri + 1 in framed and regions[ri + 1][0] - e <= self.attach_gap:
                near.add(framed[ri + 1][0][2])
            if len(near) == 1:
                turns.append((s, e, near.pop()))
        return Diarization(relabel_by_duration(merge_turns(turns)), [], self.name)


_pyannote = None
_pyannote_failed: str | None = None


def get_diarizer(settings):
    """pyannote if available, else the clustering fallback (None if no embeddings either)."""
    global _pyannote, _pyannote_failed
    cfg = settings.diarization
    if cfg.engine == "pyannote" and _pyannote is None and _pyannote_failed is None:
        if not has_module("pyannote.audio"):
            _pyannote_failed = "pyannote.audio not installed"
        else:
            try:
                _pyannote = PyannoteDiarizer(cfg.model, resolve_device(settings.device))
            except Exception as exc:  # noqa: BLE001 - gated / offline / incompatible
                msg = str(exc)
                if "gated" in msg.lower() or "403" in msg or "401" in msg:
                    msg = f"no access to {cfg.model}; accept its terms on huggingface.co with the HF_TOKEN account"
                _pyannote_failed = msg[:300]
        if _pyannote_failed:
            log.warning("pyannote diarization unavailable (%s); using embedding clustering", _pyannote_failed)
    if cfg.engine == "pyannote" and _pyannote is not None:
        engines.set_status("diarization", _pyannote.name)
        return _pyannote
    from .speakers import get_embedder

    embedder = get_embedder(settings)
    if embedder is None:
        engines.set_status("diarization", "unavailable (no pyannote, no speaker embeddings)")
        return None
    why = f" (pyannote: {_pyannote_failed})" if cfg.engine == "pyannote" else ""
    engines.set_status("diarization", f"clustering fallback{why}")
    return ClusteringDiarizer(embedder, cfg.cluster_threshold)


def overlap_seconds(start: float, end: float, overlaps: list[tuple[float, float]]) -> float:
    return sum(max(0.0, min(end, b) - max(start, a)) for a, b in overlaps)
