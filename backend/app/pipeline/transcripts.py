"""How ASR results land on a clip (shared by processing and the re-transcribe job).

Rules
* The default editable transcript is the normalized output of the configured
  source (asr.transcript_source, default: primary).
* `clip.text` is only replaced while the clip is still pending AND the text
  was never edited by a human. Approved / rejected clips keep their transcript and
  edited transcripts are never touched; an empty transcript is always filled.
* Flags are recomputed per group (primary-ASR heuristics + text normalization;
  agreement), so quality / segmentation flags survive a re-transcription.
* needs_review follows the flags for pending clips only; a reviewer's decision
  on approved / rejected clips is left alone.
"""
from __future__ import annotations

from ..models import Clip
from .asr import AsrResult, review_flags
from .text_norm import TEXT_FLAGS, comparison_form, normalize_transcript
from .wer import wer

PRIMARY_FLAGS = {"low_asr_confidence", "asr_no_speech", "asr_repetition", "empty_transcript", "sparse_transcript",
                 "no_asr"}
AGREEMENT_FLAGS = {"asr_disagreement"}


def agreement_wer(primary: str | None, secondary: str | None, language: str) -> float:
    return wer(comparison_form(primary, language), comparison_form(secondary, language))


def default_transcript(clip: Clip, settings) -> str:
    """The text a never-edited clip should show, given the current ASR outputs."""
    source = settings.asr.source_for(clip.language)
    raw = clip.asr_secondary_text if source == "secondary" and clip.asr_secondary_text else clip.asr_primary_text
    return normalize_transcript(raw, clip.language, settings).text


def _may_rewrite_text(clip: Clip) -> bool:
    """Never touch human edits; replace existing text only on pending clips; always fill an empty one
    (so an auto-rejected too_short / low_quality clip still has a transcript if someone approves it)."""
    if clip.text_edited:
        return False
    return clip.status == "pending" or not (clip.text or "").strip()


def _refresh_review(clip: Clip) -> None:
    if clip.status == "pending":
        clip.needs_review = bool(clip.flags())


def apply_primary(clip: Clip, result: AsrResult, engine_name: str, settings) -> None:
    clip.asr_primary_text = result.text
    clip.asr_confidence = result.avg_logprob
    clip.asr_primary_engine = engine_name
    norm = normalize_transcript(result.text, clip.language, settings)
    flags = set(clip.flags()) - PRIMARY_FLAGS - TEXT_FLAGS
    flags |= set(review_flags(result, clip.duration, settings.asr.review)) | norm.flags
    if _may_rewrite_text(clip):
        clip.text = default_transcript(clip, settings)
    clip.set_flags(flags)
    _refresh_review(clip)


def apply_no_primary(clip: Clip) -> None:
    clip.asr_primary_text = ""
    clip.asr_primary_engine = None
    clip.set_flags(set(clip.flags()) | {"no_asr"})
    _refresh_review(clip)


def apply_secondary(clip: Clip, text: str, engine_name: str, settings) -> None:
    clip.asr_secondary_text = text
    clip.asr_secondary_engine = engine_name
    score = agreement_wer(clip.asr_primary_text, text, clip.language)
    clip.asr_agreement = round(score, 4)
    flags = set(clip.flags()) - AGREEMENT_FLAGS
    if score > settings.asr.agreement.threshold(clip.language):
        flags.add("asr_disagreement")
    if settings.asr.source_for(clip.language) == "secondary":
        flags = (flags - TEXT_FLAGS) | normalize_transcript(text, clip.language, settings).flags
        if _may_rewrite_text(clip):
            clip.text = default_transcript(clip, settings)
    clip.set_flags(flags)
    _refresh_review(clip)
