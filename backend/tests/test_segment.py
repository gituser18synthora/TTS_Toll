"""Segmentation: duration bounds, cuts only at silence, coverage, forced splits."""
from __future__ import annotations

import numpy as np
import pytest

from app.pipeline.segment import (PlannedClip, Region, SegParams, drop_isolated_blips, plan_clips, probs_to_regions,
                                  segment, split_long_regions)

FRAME = 0.032
P = SegParams()


def probs_from_spans(spans: list[tuple[float, float]], total: float, frame: float = FRAME) -> np.ndarray:
    probs = np.full(int(np.ceil(total / frame)), 0.02, dtype=np.float32)
    for a, b in spans:
        probs[int(round(a / frame)):int(round(b / frame))] = 0.95
    return probs


def regions_from_layout(layout: list[tuple[str, float]]) -> tuple[list[Region], float]:
    regions, t = [], 0.0
    for kind, d in layout:
        if kind == "s":
            regions.append(Region(t, t + d))
        t += d
    return regions, t


def assert_cuts_in_silence(clips: list[PlannedClip], regions: list[Region]) -> None:
    """No clip boundary falls strictly inside a speech region (unless it is a forced cut)."""
    for c in clips:
        for r in regions:
            for edge, forced in ((c.speech_start, c.forced_start), (c.speech_end, c.forced_end)):
                if r.start + 1e-6 < edge < r.end - 1e-6:
                    assert forced, f"clip edge {edge:.2f} cuts speech region {r.start:.2f}-{r.end:.2f}"


def assert_full_coverage(clips: list[PlannedClip], regions: list[Region]) -> None:
    """Every bit of speech belongs to exactly one clip, in order."""
    assert clips[0].speech_start == pytest.approx(regions[0].start)
    assert clips[-1].speech_end == pytest.approx(regions[-1].end)
    for a, b in zip(clips, clips[1:]):
        assert a.speech_end <= b.speech_start + 1e-9


# ---------------------------------------------------------------------------
def test_hysteresis_regions_match_speech_spans():
    spans = [(1.0, 3.0), (3.6, 5.0), (8.0, 9.5)]
    regions = probs_to_regions(probs_from_spans(spans, 11.0), FRAME, P)
    assert len(regions) == 3
    for r, (a, b) in zip(regions, spans):
        assert abs(r.start - a) <= FRAME and abs(r.end - b) <= FRAME


def test_short_pause_does_not_split_region():
    # 0.064 s dip is shorter than min_silence (0.15 s) -> one region
    regions = probs_to_regions(probs_from_spans([(1.0, 2.0), (2.064, 4.0)], 5.0), FRAME, P)
    assert len(regions) == 1


def test_sentence_layout_respects_bounds_and_cuts_at_pauses():
    layout = [("x", 0.5)]
    rng = np.random.default_rng(1)
    for _ in range(40):
        layout += [("s", float(rng.uniform(0.6, 4.5))), ("x", float(rng.choice([0.2, 0.3, 0.6, 0.9])))]
    regions, total = regions_from_layout(layout)
    clips = plan_clips(regions, total, P)
    assert_cuts_in_silence(clips, regions)
    assert_full_coverage(clips, regions)
    for c in clips:
        assert P.min_duration <= c.duration <= P.hard_max_duration
    # most clips should land inside the preferred window
    assert sum(c.duration <= P.max_duration for c in clips) / len(clips) > 0.9


def test_prefers_long_pauses_as_cut_points():
    # 3 s + 0.2 s pause + 3 s + 0.8 s pause + 3 s + 0.2 s pause + 3 s  ->  cut at the 0.8 s pause
    regions, total = regions_from_layout([("s", 3), ("x", 0.2), ("s", 3), ("x", 0.8), ("s", 3), ("x", 0.2), ("s", 3)])
    clips = plan_clips(regions, total, P)
    assert len(clips) == 2
    assert clips[0].speech_end == pytest.approx(6.2)
    assert clips[1].speech_start == pytest.approx(7.0)


def test_never_merges_across_long_pause():
    regions, total = regions_from_layout([("s", 1.0), ("x", 3.0), ("s", 5.0)])
    clips = plan_clips(regions, total, P)
    assert len(clips) == 2
    assert clips[0].duration < P.min_duration  # kept (as too_short) rather than merged across 3 s of silence


def test_padding_is_added_but_limited_by_neighbours():
    regions, total = regions_from_layout([("x", 1.0), ("s", 5.0), ("x", 0.1), ("s", 5.0), ("x", 1.0)])
    p = SegParams(padding=0.15)
    clips = plan_clips(regions, total, p)
    assert clips[0].start == pytest.approx(1.0 - 0.15)
    assert clips[-1].end == pytest.approx(11.1 + 0.15)


def test_long_continuous_speech_is_force_split_and_flagged():
    total = 45.0
    probs = probs_from_spans([(1.0, 44.0)], total)
    probs[int(20.0 / FRAME)] = 0.55  # a shallow dip the splitter should prefer
    clips = segment(probs, FRAME, total, P)
    assert len(clips) >= 3
    for c in clips:
        assert c.duration <= P.hard_max_duration + 2 * P.padding
    assert any("forced_cut" in c.flags for c in clips)
    assert any(abs(c.speech_end - 20.0) < 0.1 for c in clips[:-1])


def test_split_long_regions_keeps_parts_within_max():
    parts = split_long_regions([Region(0.0, 50.0)], None, FRAME, P)
    assert all(r.length <= P.max_duration + 1e-9 for r in parts)
    assert all(r.length >= P.min_duration - 1e-9 for r in parts)
    assert parts[0].start == 0.0 and parts[-1].end == 50.0
    assert [r.forced_end for r in parts[:-1]] == [True] * (len(parts) - 1)


def test_isolated_blips_dropped_but_short_words_near_speech_kept():
    regions = [Region(0.0, 3.0), Region(3.3, 3.45), Region(9.0, 9.1), Region(12.0, 15.0)]
    kept = drop_isolated_blips(regions, P)
    assert Region(3.3, 3.45) in kept      # short word close to other speech
    assert Region(9.0, 9.1) not in kept   # click in the middle of silence


@pytest.mark.parametrize("seed", range(25))
def test_random_layouts_invariants(seed):
    rng = np.random.default_rng(seed)
    layout = [("x", float(rng.uniform(0, 1)))]
    for _ in range(int(rng.integers(5, 60))):
        layout += [("s", float(rng.choice([rng.uniform(0.1, 1.0), rng.uniform(1, 8), rng.uniform(8, 30)], p=[0.3, 0.6, 0.1]))),
                   ("x", float(rng.choice([rng.uniform(0.16, 0.5), rng.uniform(0.5, 1.2), rng.uniform(1.2, 4)])))]
    spans, t = [], 0.0
    for kind, d in layout:
        if kind == "s":
            spans.append((t, t + d))
        t += d
    probs = probs_from_spans(spans, t)
    clips = segment(probs, FRAME, t, P)
    regions = drop_isolated_blips(probs_to_regions(probs, FRAME, P), P)
    assert clips, "speech present but no clips"
    assert_cuts_in_silence(clips, regions)
    for a, b in zip(clips, clips[1:]):
        assert a.start < b.start and a.speech_end <= b.speech_start + 1e-9
    for c in clips:
        assert 0.0 <= c.start < c.end <= t + 1e-6
        assert c.duration <= P.hard_max_duration + 2 * P.padding
