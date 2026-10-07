"""Speech segmentation: VAD frame probabilities -> clip boundaries.

Pure functions (numpy only) so they are unit-testable without models.

1. `probs_to_regions`  hysteresis thresholding (same state machine as Silero's
                        get_speech_timestamps) -> speech regions separated by
                        pauses of at least `min_silence`.
2. `drop_isolated_blips` removes very short regions that are far from any
                        other speech (clicks, coughs).
3. `split_long_regions` a region longer than `max_duration` has no usable pause;
                        it gets candidate split points at the deepest dips of
                        the (smoothed) speech probability. Those boundaries are
                        "forced" and expensive to use.
   (multi-speaker files: `label_regions` first splits speech at speaker turns)
4. `plan_clips`         dynamic programming over region boundaries. Clips only
                        ever start/end at a pause (or a forced split), so a
                        word is never cut. The cost function prefers
                          - durations within [min_duration, max_duration]
                            (never above hard_max_duration),
                          - cutting at long pauses rather than short ones,
                          - no long pauses inside a clip (hard limit
                            `max_pause_in_clip`),
                          - clip lengths near `target_duration` (weak).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# --- cost weights (tuned so natural pauses dominate the target-length preference)
C_UNDER, W_UNDER = 10.0, 10.0      # clip shorter than min_duration: base + per second
C_OVER, W_OVER = 5.0, 2.0          # clip longer than max_duration
C_HARD = 1e6                       # longer than hard_max_duration
W_TARGET = 0.1                     # per second away from target_duration
STRONG_PAUSE = 0.5                 # pauses >= this are "free" cut points
W_WEAK_CUT = 8.0                   # per second the pause is shorter than STRONG_PAUSE
C_FORCED_CUT = 15.0                # cutting inside continuous speech
SOFT_INTERNAL_PAUSE = 0.7          # internal pauses longer than this cost a bit
W_INTERNAL_PAUSE = 4.0


@dataclass(frozen=True)
class SegParams:
    min_duration: float = 2.0
    max_duration: float = 15.0
    hard_max_duration: float = 20.0
    target_duration: float = 8.0
    padding: float = 0.15
    max_pause_in_clip: float = 1.2
    threshold: float = 0.5
    neg_threshold: float = 0.35
    min_speech: float = 0.25
    min_silence: float = 0.15

    @classmethod
    def from_settings(cls, settings) -> "SegParams":
        s, v = settings.segmentation, settings.vad
        return cls(
            min_duration=s.min_duration,
            max_duration=s.max_duration,
            hard_max_duration=s.hard_max_duration,
            target_duration=s.target_duration,
            padding=s.padding_ms / 1000.0,
            max_pause_in_clip=s.max_pause_in_clip,
            threshold=v.threshold,
            neg_threshold=v.exit_threshold,
            min_speech=v.min_speech_ms / 1000.0,
            min_silence=v.min_silence_ms / 1000.0,
        )


@dataclass
class Region:
    start: float
    end: float
    forced_end: bool = False  # boundary after this region is a split inside continuous speech
    label: str | None = None  # diarization speaker (multi-speaker files); clips never mix labels

    @property
    def length(self) -> float:
        return self.end - self.start


@dataclass
class PlannedClip:
    start: float          # including padding
    end: float
    speech_start: float   # first/last VAD speech inside the clip
    speech_end: float
    forced_start: bool = False
    forced_end: bool = False
    flags: list[str] = field(default_factory=list)
    label: str | None = None

    @property
    def duration(self) -> float:
        return self.end - self.start


# ---------------------------------------------------------------------------
# 1. probabilities -> regions
# ---------------------------------------------------------------------------
def probs_to_regions(probs: np.ndarray, frame_sec: float, p: SegParams) -> list[Region]:
    """Hysteresis: enter speech at >= threshold, leave after `min_silence` below neg_threshold."""
    min_silence_frames = max(1, int(round(p.min_silence / frame_sec)))
    regions: list[Region] = []
    triggered = False
    start = 0
    temp_end = -1
    for i, prob in enumerate(np.asarray(probs, dtype=np.float32)):
        if prob >= p.threshold:
            temp_end = -1
            if not triggered:
                triggered, start = True, i
            continue
        if triggered and prob < p.neg_threshold:
            if temp_end < 0:
                temp_end = i
            if i - temp_end + 1 >= min_silence_frames:
                regions.append(Region(start * frame_sec, temp_end * frame_sec))
                triggered, temp_end = False, -1
    if triggered:
        end = temp_end if temp_end >= 0 else len(probs)
        regions.append(Region(start * frame_sec, end * frame_sec))
    return [r for r in regions if r.length > 0]


def drop_isolated_blips(regions: list[Region], p: SegParams) -> list[Region]:
    """Remove regions shorter than min_speech whose nearest neighbour is further than max_pause_in_clip."""
    if not regions:
        return []
    keep: list[Region] = []
    for i, r in enumerate(regions):
        if r.length >= p.min_speech:
            keep.append(r)
            continue
        gap_before = r.start - regions[i - 1].end if i > 0 else float("inf")
        gap_after = regions[i + 1].start - r.end if i + 1 < len(regions) else float("inf")
        if min(gap_before, gap_after) <= p.max_pause_in_clip:
            keep.append(r)
    return keep


# ---------------------------------------------------------------------------
# 2. split regions that have no pause at all
# ---------------------------------------------------------------------------
def _smooth(x: np.ndarray, width: int) -> np.ndarray:
    if width <= 1 or len(x) < width:
        return x
    kernel = np.ones(width, dtype=np.float32) / width
    return np.convolve(x, kernel, mode="same")


def split_long_regions(regions: list[Region], probs: np.ndarray | None, frame_sec: float,
                       p: SegParams) -> list[Region]:
    """Recursively bisect regions longer than max_duration at the deepest probability dip.

    The split point keeps both halves >= min_duration and is biased slightly
    toward the middle. Resulting inner boundaries are marked `forced_end`.
    """
    smooth = _smooth(np.asarray(probs, dtype=np.float32), 3) if probs is not None and len(probs) else None

    def split(r: Region) -> list[Region]:
        if r.length <= p.max_duration:
            return [r]
        lo, hi = r.start + p.min_duration, r.end - p.min_duration
        if hi <= lo:  # pathological config; fall back to the middle
            lo = hi = (r.start + r.end) / 2
        mid = (r.start + r.end) / 2
        t = mid
        if smooth is not None:
            i0, i1 = int(np.ceil(lo / frame_sec)), int(np.floor(hi / frame_sec))
            i1 = min(i1, len(smooth) - 1)
            if i1 >= i0:
                idx = np.arange(i0, i1 + 1)
                times = idx * frame_sec
                half = max((hi - lo) / 2, 1e-6)
                score = smooth[idx] + 0.1 * np.abs(times - mid) / half
                t = float(times[int(np.argmin(score))])
        left = Region(r.start, t, forced_end=True, label=r.label)
        right = Region(t, r.end, forced_end=r.forced_end, label=r.label)
        return split(left) + split(right)

    out: list[Region] = []
    for r in regions:
        out.extend(split(r))
    return out


# ---------------------------------------------------------------------------
# 3. dynamic-programming clip plan
# ---------------------------------------------------------------------------
def _duration_cost(d: float, p: SegParams) -> float:
    cost = W_TARGET * abs(d - p.target_duration)
    if d > p.hard_max_duration:
        return cost + C_HARD
    if d > p.max_duration:
        cost += C_OVER + W_OVER * (d - p.max_duration)
    elif d < p.min_duration:
        cost += C_UNDER + W_UNDER * (p.min_duration - d)
    return cost


def _cut_cost(gap: float, forced: bool) -> float:
    if forced:
        return C_FORCED_CUT
    return W_WEAK_CUT * max(0.0, STRONG_PAUSE - gap)


def _internal_pause_cost(gap: float, forced: bool) -> float:
    if forced:
        return 0.0
    return W_INTERNAL_PAUSE * max(0.0, gap - SOFT_INTERNAL_PAUSE)


DROPPED = "<dropped>"  # label of speech no clip may contain


def label_regions(regions: list[Region], turns: list[tuple[float, float, str]], keep: set[str] | None = None,
                  edge_tolerance: float = 0.2, stats: dict | None = None) -> list[Region]:
    """Split speech regions at speaker turns (multi-speaker files).

    Speech of labels not in `keep` (ignored speakers) and speech the diarization attributed to nobody
    become DROPPED regions instead of vanishing: plan_clips treats them like another speaker - no clip
    spans one, padding stops halfway into the gap before it - and segment() discards their clips.
    (Removing them outright would let a neighbouring clip merge across, e.g., an ignored "haan".)
    Unattributed slivers up to `edge_tolerance` that touch a turn inside the same VAD region are
    boundary jitter between the VAD and the diarizer: they join that turn, so onsets are not clipped.
    """
    import bisect

    turns = sorted(turns)
    starts = [t[0] for t in turns]
    ignored = unattributed = 0.0
    out: list[Region] = []
    for r in regions:
        pieces: list[tuple[float, float, str | None]] = []  # covers r: turns and the gaps between them
        pos = r.start
        k = max(0, bisect.bisect_right(starts, r.start) - 1)
        while k < len(turns) and turns[k][0] < r.end:
            ts, te, label = turns[k]
            a, b = max(pos, ts), min(r.end, te)
            if b > a:
                if a > pos:
                    pieces.append((pos, a, None))
                pieces.append((a, b, label))
                pos = b
            k += 1
        if r.end > pos:
            pieces.append((pos, r.end, None))
        labels = [pc[2] for pc in pieces]
        for i, (a, b, label) in enumerate(pieces):
            if label is not None or b - a > edge_tolerance:
                continue
            prev = pieces[i - 1][2] if i > 0 else None
            nxt = pieces[i + 1][2] if i + 1 < len(pieces) else None
            if i == 0 and nxt is not None:
                labels[i] = nxt  # before the first turn of this region
            elif i == len(pieces) - 1 and prev is not None:
                labels[i] = prev  # after the last one
            elif prev is not None and prev == nxt:
                labels[i] = prev  # a hole inside one speaker's turn
        for (a, b, _), label in zip(pieces, labels):
            if label is None:
                unattributed += b - a
                label = DROPPED
            elif keep is not None and label not in keep:
                ignored += b - a
                label = DROPPED
            if out and out[-1].label == label and a - out[-1].end < 1e-6:
                out[-1] = Region(out[-1].start, b, label=label)
            else:
                out.append(Region(a, b, label=label))
    if stats is not None:
        stats.update(ignored_sec=round(ignored, 2), unattributed_sec=round(unattributed, 2))
    return out


def plan_clips(regions: list[Region], total_duration: float, p: SegParams) -> list[PlannedClip]:
    n = len(regions)
    if n == 0:
        return []
    gaps = [regions[k + 1].start - regions[k].end for k in range(n - 1)]
    forced = [regions[k].forced_end for k in range(n - 1)]
    labels = [r.label for r in regions]
    change = [labels[k] != labels[k + 1] for k in range(n - 1)]  # speaker change: free cut, never merged

    def pad_left(i: int) -> float:
        if i == 0:
            return max(0.0, min(p.padding, regions[0].start))
        if forced[i - 1]:
            return 0.0
        room = gaps[i - 1] / 2 if change[i - 1] else gaps[i - 1]  # never pad into the other speaker
        return max(0.0, min(p.padding, room))

    def pad_right(j: int) -> float:
        if j == n - 1:
            return max(0.0, min(p.padding, total_duration - regions[j].end))
        if forced[j]:
            return 0.0
        room = gaps[j] / 2 if change[j] else gaps[j]
        return max(0.0, min(p.padding, room))

    inf = float("inf")
    best = [inf] * (n + 1)
    back = [-1] * (n + 1)
    best[0] = 0.0
    for j in range(n):
        cut = 0.0 if j == n - 1 or change[j] else _cut_cost(gaps[j], forced[j])
        internal = 0.0
        for i in range(j, -1, -1):
            if i < j:
                if change[i]:
                    break  # a clip never spans two speakers
                g, f = gaps[i], forced[i]
                if not f and g > p.max_pause_in_clip:
                    break  # never merge across a long pause
                internal += _internal_pause_cost(g, f)
                if regions[j].end - regions[i].start > p.hard_max_duration:
                    break
            if best[i] == inf:
                continue
            dur = regions[j].end - regions[i].start + pad_left(i) + pad_right(j)
            cost = best[i] + internal + _duration_cost(dur, p) + cut
            if cost < best[j + 1]:
                best[j + 1], back[j + 1] = cost, i

    spans: list[tuple[int, int]] = []
    j = n
    while j > 0:
        i = back[j]
        spans.append((i, j - 1))
        j = i
    spans.reverse()

    clips: list[PlannedClip] = []
    for i, j in spans:
        c = PlannedClip(
            start=regions[i].start - pad_left(i),
            end=regions[j].end + pad_right(j),
            speech_start=regions[i].start,
            speech_end=regions[j].end,
            forced_start=i > 0 and forced[i - 1],
            forced_end=j < n - 1 and forced[j],
            label=regions[i].label,
        )
        if c.forced_start or c.forced_end:
            c.flags.append("forced_cut")
        clips.append(c)
    return clips


def segment(probs: np.ndarray, frame_sec: float, total_duration: float, p: SegParams,
            turns: list[tuple[float, float, str]] | None = None, keep: set[str] | None = None,
            stats: dict | None = None) -> list[PlannedClip]:
    regions = probs_to_regions(probs, frame_sec, p)
    # the last VAD frame is zero-padded, so clamp to the real audio length
    regions = [Region(r.start, min(r.end, total_duration)) for r in regions if r.start < total_duration]
    if turns is not None:  # multi-speaker file: one speaker per clip
        regions = label_regions(regions, turns, keep, stats=stats)
    regions = drop_isolated_blips(regions, p)
    regions = split_long_regions(regions, probs, frame_sec, p)
    return [c for c in plan_clips(regions, total_duration, p) if c.label != DROPPED]
