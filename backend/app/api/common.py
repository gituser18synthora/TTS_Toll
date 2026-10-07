"""Shared helpers for routers: dependencies and ORM -> schema converters."""
from __future__ import annotations

import json

from fastapi import HTTPException
from sqlalchemy import func, select
from sqlalchemy.orm import Session, object_session

from ..config import Settings, get_settings
from ..models import Clip, Export, Job, SourceFile, Speaker
from ..paths import DataPaths
from ..pipeline.text_norm import normalize_transcript
from ..schemas import ClipCounts, ClipOut, ExportOut, JobOut, SourceOut


def settings_dep() -> Settings:
    return get_settings()


def paths_dep() -> DataPaths:
    return DataPaths.from_settings(get_settings())


def get_or_404(db: Session, model, obj_id: int):
    obj = db.get(model, obj_id)
    if obj is None:
        raise HTTPException(404, f"{model.__name__} {obj_id} not found")
    return obj


def check_language(settings: Settings, language: str) -> str:
    if language not in settings.languages:
        raise HTTPException(422, f"unknown language '{language}'; expected one of {sorted(settings.languages)}")
    return language


def get_or_create_speaker(db: Session, name: str, gender: str) -> Speaker:
    """Speakers are unique by name (case-insensitive); gender must match an existing speaker."""
    sp = db.scalars(select(Speaker).where(func.lower(Speaker.name) == name.lower())).first()
    if sp is None:
        sp = Speaker(name=name, gender=gender)
        db.add(sp)
        db.flush()
    elif sp.gender != gender:
        raise HTTPException(409, f"speaker '{sp.name}' already exists with gender '{sp.gender}'")
    return sp


def _norm(text: str | None, language: str) -> str | None:
    return None if text is None else normalize_transcript(text, language, get_settings()).text


def clip_out(c: Clip) -> ClipOut:
    return ClipOut(
        id=c.id, source_file_id=c.source_file_id, source_filename=c.source_file.original_filename,
        denoised=bool(c.source_file.enhanced_path),
        speaker_id=c.speaker_id, speaker_name=c.speaker.name, gender=c.speaker.gender, language=c.language,
        idx=c.idx, start_time=c.start_time, end_time=c.end_time, duration=c.duration, text=c.text,
        text_edited=c.text_edited, asr_primary_text=c.asr_primary_text, asr_secondary_text=c.asr_secondary_text,
        primary_normalized=_norm(c.asr_primary_text, c.language),
        secondary_normalized=_norm(c.asr_secondary_text, c.language),
        asr_agreement=c.asr_agreement, asr_confidence=c.asr_confidence, asr_primary_engine=c.asr_primary_engine,
        asr_secondary_engine=c.asr_secondary_engine, quality_score=c.quality_score,
        quality_info=json.loads(c.quality_info) if c.quality_info else None, loudness_lufs=c.loudness_lufs,
        needs_review=c.needs_review, flags=c.flags(), status=c.status, speaker_label=c.speaker_label,
        speaker_similarity=c.speaker_similarity, duplicate_of=c.duplicate_of,
        reject_reason=c.reject_reason, reviewed_at=c.reviewed_at, updated_at=c.updated_at,
        audio_url=f"/api/clips/{c.id}/audio?h={(c.audio_hash or '')[:10]}",
    )


def job_out(j: Job | None) -> JobOut | None:
    return JobOut.model_validate(j) if j is not None else None


def clip_counts(db: Session, source_ids: list[int] | None = None) -> dict[int, ClipCounts]:
    q = select(Clip.source_file_id, Clip.status, func.count(Clip.id), func.sum(Clip.duration))
    if source_ids is not None:
        q = q.where(Clip.source_file_id.in_(source_ids))
    out: dict[int, ClipCounts] = {}
    for sid, status, n, dur in db.execute(q.group_by(Clip.source_file_id, Clip.status)):
        c = out.setdefault(sid, ClipCounts())
        c.total += n
        c.total_hours += (dur or 0) / 3600
        if status in ("pending", "approved", "rejected"):
            setattr(c, status, n)
        if status == "approved":
            c.approved_hours = (dur or 0) / 3600
    q = select(Clip.source_file_id, func.count(Clip.id)).where(Clip.needs_review.is_(True), Clip.status == "pending")
    if source_ids is not None:
        q = q.where(Clip.source_file_id.in_(source_ids))
    for sid, n in db.execute(q.group_by(Clip.source_file_id)):
        out.setdefault(sid, ClipCounts()).needs_review = n
    for c in out.values():
        c.total_hours = round(c.total_hours, 3)
        c.approved_hours = round(c.approved_hours, 3)
    return out


def latest_jobs(db: Session, *, source_ids: list[int] | None = None, export_ids: list[int] | None = None) -> dict:
    col = Job.source_file_id if source_ids is not None else Job.export_id
    ids = source_ids if source_ids is not None else export_ids
    if not ids:
        return {}
    sub = select(func.max(Job.id)).where(col.in_(ids)).group_by(col)
    return {getattr(j, col.key): j for j in db.scalars(select(Job).where(Job.id.in_(sub)))}


def _source_speaker_label(s: SourceFile) -> str:
    if not s.multi_speaker:
        return s.speaker.name if s.speaker else "?"
    ids = [m["speaker_id"] for m in s.get_speaker_map().values() if not m.get("ignore")]
    session = object_session(s)
    names = [session.get(Speaker, i).name for i in dict.fromkeys(ids)] if ids and session else []
    return ", ".join(names) or "multiple speakers"


def source_out(s: SourceFile, counts: ClipCounts | None = None, job: Job | None = None) -> SourceOut:
    return SourceOut(
        id=s.id, original_filename=s.original_filename, original_path=s.original_path, speaker_id=s.speaker_id,
        speaker_name=_source_speaker_label(s), gender=s.speaker.gender if s.speaker else None,
        multi_speaker=s.multi_speaker, language=s.language, duration=s.duration,
        status=s.status, stage=s.stage, progress=s.progress, error=s.error, notes=s.notes,
        steps_done=s.get_steps_done(), enhanced=bool(s.enhanced_path), created_at=s.created_at,
        updated_at=s.updated_at,
        counts=counts or ClipCounts(), job=job_out(job),
    )


def export_out(e: Export, paths: DataPaths, job: Job | None = None) -> ExportOut:
    zip_size = None
    if e.zip_path:
        p = paths.abs(e.zip_path)
        zip_size = p.stat().st_size if p.exists() else None
    return ExportOut(
        id=e.id, name=e.name, params=json.loads(e.params or "{}"), status=e.status, out_dir=e.out_dir,
        zip_path=e.zip_path, zip_size=zip_size, clip_count=e.clip_count, total_duration=e.total_duration,
        stats=json.loads(e.stats) if e.stats else None, error=e.error, created_at=e.created_at,
        finished_at=e.finished_at, job=job_out(job),
    )
