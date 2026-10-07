"""Duplicate clips: exact (identical PCM hash) and near-identical (same recording again).

Near-identical = same speaker, same normalized transcript (>= 3 words), length within
10 %, and the waveforms are the *same recording*: peak normalized cross-correlation at
8 kHz >= dedup.min_similarity. A re-encoded / re-normalized / slightly re-trimmed copy
scores ~0.9-1.0; a second *take* of the same sentence scores far lower, so genuine
repeated readings are kept.

Within a duplicate group the clip kept is the best-reviewed one (approved > pending >
rejected), then the oldest. Pending duplicates are rejected as `duplicate` with
`duplicate_of` pointing at the kept clip; approved / rejected ones only get the flag.
Audio is never deleted.
"""
from __future__ import annotations

import logging
from collections import defaultdict
from itertools import combinations

import numpy as np
from sqlalchemy import select

from ..db import session_scope
from ..models import Clip
from . import audio_ops
from .text_norm import comparison_form

log = logging.getLogger(__name__)

STATUS_RANK = {"approved": 0, "pending": 1, "rejected": 2}


def same_recording_score(a: np.ndarray, b: np.ndarray, sr: int, max_lag: float = 0.5) -> float:
    """Peak |normalized cross-correlation| within +/- max_lag seconds, on 8 kHz audio."""
    a8 = audio_ops.resample_poly_to(a, sr, 8000).astype(np.float64)
    b8 = audio_ops.resample_poly_to(b, sr, 8000).astype(np.float64)
    a8 -= a8.mean()
    b8 -= b8.mean()
    na, nb = np.linalg.norm(a8), np.linalg.norm(b8)
    if na == 0 or nb == 0:
        return 0.0
    n = len(a8) + len(b8) - 1
    nfft = 1 << (n - 1).bit_length()
    cc = np.fft.irfft(np.fft.rfft(a8, nfft) * np.conj(np.fft.rfft(b8, nfft)), nfft)
    lag = min(int(max_lag * 8000), nfft // 2 - 1)
    window = np.concatenate([cc[:lag + 1], cc[nfft - lag:]]) if lag > 0 else cc[:1]
    return float(np.max(np.abs(window)) / (na * nb))


def _keeper(a, b):
    """(keep, drop) between two rows with .id and .status."""
    ka = (STATUS_RANK.get(a.status, 3), a.id)
    kb = (STATUS_RANK.get(b.status, 3), b.id)
    return (a, b) if ka <= kb else (b, a)


def find_duplicates(source_id: int, paths, min_similarity: float) -> list[tuple[int, int, str]]:
    """[(keep_id, drop_id, kind)] for clips of `source_id` against the whole database."""
    with session_scope() as s:
        mine = s.execute(select(Clip.id, Clip.audio_hash, Clip.speaker_id).where(Clip.source_file_id == source_id)).all()
    if not mine:
        return []
    pairs: list[tuple[int, int, str]] = []

    hashes = sorted({r.audio_hash for r in mine if r.audio_hash})
    by_hash: dict[str, list] = defaultdict(list)
    with session_scope() as s:
        for i in range(0, len(hashes), 500):
            for r in s.execute(select(Clip.id, Clip.audio_hash, Clip.status)
                               .where(Clip.audio_hash.in_(hashes[i:i + 500]))):
                by_hash[r.audio_hash].append(r)
    for group in by_hash.values():
        if len(group) > 1:
            keep = min(group, key=lambda r: (STATUS_RANK.get(r.status, 3), r.id))
            pairs += [(keep.id, r.id, "exact") for r in group if r.id != keep.id]
    exact_dropped = {d for _, d, _ in pairs}

    audio_cache: dict[int, tuple[np.ndarray, int]] = {}

    def audio(row) -> tuple[np.ndarray, int]:
        if row.id not in audio_cache:
            audio_cache[row.id] = audio_ops.read_wav(paths.abs(row.clip_path))
        return audio_cache[row.id]

    for speaker_id in {r.speaker_id for r in mine}:
        with session_scope() as s:
            # clips already resolved as duplicates never become keepers or drops again (stable duplicate_of)
            rows = s.execute(select(Clip.id, Clip.text, Clip.duration, Clip.clip_path, Clip.status, Clip.language,
                                    Clip.source_file_id)
                             .where(Clip.speaker_id == speaker_id, Clip.text.is_not(None),
                                    Clip.duplicate_of.is_(None))).all()
        groups: dict[str, list] = defaultdict(list)
        for r in rows:
            key = comparison_form(r.text, r.language)
            if len(key.split()) >= 3:
                groups[key].append(r)
        for members in groups.values():
            if len(members) < 2 or not any(m.source_file_id == source_id for m in members):
                continue
            for a, b in combinations(members, 2):
                if source_id not in (a.source_file_id, b.source_file_id) or a.id in exact_dropped or b.id in exact_dropped:
                    continue
                if abs(a.duration - b.duration) > max(0.5, 0.1 * max(a.duration, b.duration)):
                    continue
                (xa, sra), (xb, srb) = audio(a), audio(b)
                if sra != srb:
                    xb = audio_ops.resample_poly_to(xb, srb, sra)
                if same_recording_score(xa, xb, sra) >= min_similarity:
                    keep, drop = _keeper(a, b)
                    pairs.append((keep.id, drop.id, "near"))
    return pairs


def apply_duplicates(pairs: list[tuple[int, int, str]]) -> dict:
    stats = {"exact": 0, "near": 0, "rejected": 0}
    seen: set[int] = set()
    with session_scope(write=True) as s:
        for keep_id, drop_id, kind in pairs:
            if drop_id in seen or drop_id == keep_id:
                continue
            seen.add(drop_id)
            clip = s.get(Clip, drop_id)
            if clip is None:
                continue
            clip.duplicate_of = keep_id
            clip.set_flags(set(clip.flags()) | {"duplicate"})
            if clip.status == "pending":
                clip.status, clip.reject_reason, clip.needs_review = "rejected", "duplicate", False
                stats["rejected"] += 1
            stats[kind] += 1
    return stats
