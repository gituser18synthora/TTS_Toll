"""Clip review endpoints."""
from __future__ import annotations

import io
from typing import Literal

import numpy as np
import soundfile as sf

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse, Response
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from ..db import get_db, get_db_rw
from ..models import Clip, utcnow
from ..paths import DataPaths
from ..pipeline import audio_ops
from ..config import Settings
from ..pipeline.text_norm import TEXT_FLAGS, clean_text
from ..pipeline.transcripts import default_transcript
from ..schemas import BulkAction, BulkResult, ClipOut, ClipPage, ClipUpdate
from .common import clip_out, get_or_404, paths_dep, settings_dep

router = APIRouter(tags=["clips"])

SORTS = {
    "position": None,
    "id": Clip.id,
    "quality_score": Clip.quality_score,
    "duration": Clip.duration,
    "asr_agreement": Clip.asr_agreement,
    "asr_confidence": Clip.asr_confidence,
    "updated_at": Clip.updated_at,
}


def _filtered(q, speaker_id, language, status, needs_review, source_file_id, flag, search):
    if speaker_id:
        q = q.where(Clip.speaker_id == speaker_id)
    if language:
        q = q.where(Clip.language == language)
    if status and status != "all":
        q = q.where(Clip.status == status)
    if needs_review is not None:
        q = q.where(Clip.needs_review.is_(needs_review))
    if source_file_id:
        q = q.where(Clip.source_file_id == source_file_id)
    if flag:
        q = q.where((Clip.review_reason == flag) | Clip.review_reason.like(f"%{flag}%"))
    if search:
        q = q.where(Clip.text.like(f"%{search}%"))
    return q


@router.get("/clips", response_model=ClipPage)
def list_clips(
    speaker_id: int | None = None,
    language: str | None = None,
    status: Literal["pending", "approved", "rejected", "all"] | None = "pending",
    needs_review: bool | None = None,
    source_file_id: int | None = None,
    flag: str | None = None,
    q: str | None = Query(default=None, max_length=200),
    sort: Literal["position", "id", "quality_score", "duration", "asr_agreement", "asr_confidence",
                  "updated_at"] = "position",
    order: Literal["asc", "desc"] = "asc",
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=50, ge=1, le=500),
    offset: int | None = Query(default=None, ge=0, description="overrides page (lets the UI continue after reviewed rows drop out of the filter)"),
    db: Session = Depends(get_db),
) -> ClipPage:
    args = (speaker_id, language, status, needs_review, source_file_id, flag, q)
    total = db.scalar(_filtered(select(func.count(Clip.id)), *args)) or 0
    stmt = _filtered(select(Clip), *args)
    col = SORTS[sort]
    if col is not None:
        key = col.desc() if order == "desc" else col.asc()
        stmt = stmt.order_by(key.nulls_last(), Clip.source_file_id, Clip.idx)
    elif order == "desc":
        stmt = stmt.order_by(Clip.source_file_id.desc(), Clip.idx.desc())
    else:
        stmt = stmt.order_by(Clip.source_file_id, Clip.idx)
    start = offset if offset is not None else (page - 1) * page_size
    clips = db.scalars(stmt.offset(start).limit(page_size)).unique().all()
    return ClipPage(items=[clip_out(c) for c in clips], total=total, page=start // page_size + 1,
                    page_size=page_size, offset=start)


@router.get("/clips/{clip_id}", response_model=ClipOut)
def get_clip(clip_id: int, db: Session = Depends(get_db)) -> ClipOut:
    return clip_out(get_or_404(db, Clip, clip_id))


def _apply_status(clip: Clip, status: str, reject_reason: str | None) -> None:
    clip.status = status
    if status == "approved":
        clip.reject_reason = None
        clip.reviewed_at = utcnow()
        clip.needs_review = False
    elif status == "rejected":
        clip.reject_reason = reject_reason or clip.reject_reason or "manual"
        clip.reviewed_at = utcnow()
        clip.needs_review = False
    else:  # back to pending: auto flags apply again
        clip.reject_reason = None
        clip.reviewed_at = None
        clip.needs_review = bool(clip.flags())


@router.patch("/clips/{clip_id}", response_model=ClipOut)
def update_clip(clip_id: int, body: ClipUpdate, db: Session = Depends(get_db_rw),
                settings: Settings = Depends(settings_dep)) -> ClipOut:
    """Edit transcript and/or review status. The edited text is what gets exported."""
    clip = get_or_404(db, Clip, clip_id)
    if body.text is not None:
        new = clean_text(body.text)
        if new != (clip.text or ""):
            clip.text = new
            clip.text_edited = new != default_transcript(clip, settings)
            if clip.text_edited:  # those flags described the machine transcript, not this one
                clip.set_flags(set(clip.flags()) - TEXT_FLAGS)
                if clip.status == "pending":
                    clip.needs_review = bool(clip.flags())
    if body.status is not None:
        if body.status == "approved" and not (clip.text or "").strip():
            raise HTTPException(422, "cannot approve a clip with an empty transcript")
        _apply_status(clip, body.status, body.reject_reason)
    elif body.reject_reason is not None and clip.status == "rejected":
        clip.reject_reason = body.reject_reason
    if body.needs_review is not None:
        clip.needs_review = body.needs_review
    db.commit()
    db.refresh(clip)
    return clip_out(clip)


@router.post("/clips/bulk", response_model=BulkResult)
def bulk_update(body: BulkAction, db: Session = Depends(get_db_rw)) -> BulkResult:
    status = {"approve": "approved", "reject": "rejected", "pending": "pending"}[body.action]
    clips = db.scalars(select(Clip).where(Clip.id.in_(body.ids))).unique().all()
    updated = skipped = 0
    for clip in clips:
        if status == "approved" and not (clip.text or "").strip():
            skipped += 1
            continue
        _apply_status(clip, status, body.reject_reason)
        updated += 1
    db.commit()
    return BulkResult(updated=updated, skipped=skipped,
                      skipped_reason="empty transcript" if skipped else None)


@router.get("/clips/{clip_id}/audio")
def clip_audio(clip_id: int, variant: Literal["clip", "original"] = "clip", db: Session = Depends(get_db),
               paths: DataPaths = Depends(paths_dep), settings: Settings = Depends(settings_dep)) -> Response:
    """The clip as exported, or (variant=original) the same span from the un-denoised audio for A/B listening."""
    clip = get_or_404(db, Clip, clip_id)
    if variant == "clip":
        path = paths.abs(clip.clip_path)
        if not path.exists():
            raise HTTPException(404, "clip audio missing on disk")
        return FileResponse(path, media_type="audio/wav", headers={"Cache-Control": "private, max-age=86400"})
    src = clip.source_file
    if not src.enhanced_path or not src.converted_path or not paths.abs(src.converted_path).exists():
        raise HTTPException(404, "this file was not denoised, so there is no separate original")
    x, sr = audio_ops.read_segment(paths.abs(src.converted_path), clip.start_time, clip.end_time)
    lufs = audio_ops.measure_lufs(x, sr)
    if lufs is not None:  # level-match with the processed clip for a fair comparison
        x = x * 10 ** (min(settings.clip.max_gain_db, settings.clip.target_lufs - lufs) / 20)
    x, _ = audio_ops.limit_peaks(x.astype(np.float32), sr, ceiling_db=settings.clip.peak_dbfs)
    buf = io.BytesIO()
    sf.write(buf, audio_ops.to_pcm16(x), sr, format="WAV", subtype="PCM_16")
    return Response(buf.getvalue(), media_type="audio/wav", headers={"Cache-Control": "private, max-age=3600"})
