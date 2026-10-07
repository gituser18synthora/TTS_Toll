"""Health, UI config and summary statistics."""
from __future__ import annotations

import json
import os
from collections import Counter
from datetime import timedelta

from fastapi import APIRouter, Depends
from sqlalchemy import case, func, select
from sqlalchemy.orm import Session

from .. import __version__
from ..config import Settings
from ..db import get_db
from ..devices import capability_report
from ..models import Clip, Job, SourceFile, Speaker, WorkerInfo, utcnow
from .common import settings_dep

router = APIRouter(tags=["system"])

_caps: dict | None = None


@router.get("/health")
def health(db: Session = Depends(get_db), settings: Settings = Depends(settings_dep)) -> dict:
    global _caps
    if _caps is None:
        _caps = capability_report(settings)
    cutoff = utcnow() - timedelta(seconds=45)
    workers = db.scalars(select(WorkerInfo).where(WorkerInfo.heartbeat_at >= cutoff)).all()
    engines: dict = {}
    for w in workers:
        try:
            engines.update(json.loads(w.info or "{}").get("engines", {}))
        except ValueError:
            pass
    return {
        "status": "ok",
        "version": __version__,
        "worker": {"online": bool(workers), "count": len(workers),
                   "devices": sorted({w.device or "?" for w in workers}), "engines": engines},
        "capabilities": {**_caps, "hf_token": bool(os.environ.get("HF_TOKEN"))},
        "data_root": str(settings.data_root),
    }


@router.get("/config")
def ui_config(settings: Settings = Depends(settings_dep)) -> dict:
    return {
        "languages": [{"code": k, "name": v.name, "whisper": v.whisper} for k, v in settings.languages.items()],
        "genders": ["male", "female"],
        "allowed_extensions": settings.upload.allowed_extensions,
        "max_file_gb": settings.upload.max_file_gb,
        "segmentation": settings.segmentation.model_dump(),
        "clip": settings.clip.model_dump(),
        "sample_rate": settings.audio.sample_rate,
        "roman_scheme": settings.text.roman_scheme,
        "romanizable_languages": ["hi", "hinglish"],
        "asr": {"engine": settings.asr.primary.engine, "model_cpu": settings.asr.primary.model_cpu,
                "model_gpu": settings.asr.primary.model_gpu,
                "secondary": settings.asr.secondary.model_dump(),
                "agreement_max_wer": settings.asr.agreement.max_wer},
        "quality": {"engine": settings.quality.engine, "reject_below": settings.quality.reject_below,
                    "review_below": settings.quality.review_below},
        "denoise": settings.denoise.engine,
        "reject_reasons": ["manual", "bad_audio", "noise", "music", "overlap", "wrong_speaker",
                           "bad_transcript", "mispronounced", "low_quality", "too_short", "too_long", "other"],
    }


def _breakdown(db: Session, group_col) -> dict:
    rows = db.execute(select(group_col, Clip.status, func.count(Clip.id), func.sum(Clip.duration))
                      .group_by(group_col, Clip.status)).all()
    out: dict = {}
    for key, status, n, dur in rows:
        entry = out.setdefault(key, {"pending": 0.0, "approved": 0.0, "rejected": 0.0, "clips": 0})
        entry[status] = round((dur or 0) / 3600, 4)
        entry["clips"] += n
    return out


@router.get("/stats")
def stats(db: Session = Depends(get_db)) -> dict:
    """Hours per speaker / language / status, review backlog, job activity."""
    by_status = {s: {"clips": n, "hours": round((d or 0) / 3600, 4)} for s, n, d in db.execute(
        select(Clip.status, func.count(Clip.id), func.sum(Clip.duration)).group_by(Clip.status))}
    names = dict(db.execute(select(Speaker.id, Speaker.name)).all())
    per_speaker = {names.get(k, str(k)): v for k, v in _breakdown(db, Clip.speaker_id).items()}
    needs_review = db.scalar(select(func.count(Clip.id)).where(Clip.needs_review.is_(True),
                                                               Clip.status == "pending")) or 0
    avg_quality = db.scalar(select(func.avg(Clip.quality_score)))
    jobs = dict(db.execute(select(Job.status, func.count(Job.id))
                           .where(Job.status.in_(("queued", "running"))).group_by(Job.status)).all())
    sources = dict(db.execute(select(SourceFile.status, func.count(SourceFile.id)).group_by(SourceFile.status)).all())
    raw_hours = db.scalar(select(func.sum(SourceFile.duration))) or 0
    edited = db.scalar(select(func.sum(case((Clip.text_edited.is_(True), 1), else_=0)))) or 0
    genders = dict(db.execute(select(Speaker.name, Speaker.gender)).all())

    # quality histogram (0.25-wide bins over 1-5), per status
    bins = [round(1 + 0.25 * i, 2) for i in range(17)]
    qhist = {st: [0] * 16 for st in ("approved", "pending", "rejected")}
    for score, status in db.execute(select(Clip.quality_score, Clip.status).where(Clip.quality_score.is_not(None))):
        k = min(15, max(0, int((score - 1) / 0.25)))
        if status in qhist:
            qhist[status][k] += 1

    # flags on pending clips + reasons for rejection
    flag_counts: Counter = Counter()
    for (reason,) in db.execute(select(Clip.review_reason).where(Clip.status == "pending", Clip.review_reason.is_not(None))):
        flag_counts.update(f for f in reason.split(",") if f)
    reject_reasons = dict(db.execute(select(Clip.reject_reason, func.count(Clip.id))
                                     .where(Clip.status == "rejected").group_by(Clip.reject_reason)).all())

    # review throughput: clips reviewed per day (last 30 days)
    since = utcnow() - timedelta(days=30)
    per_day = db.execute(select(func.date(Clip.reviewed_at), Clip.status, func.count(Clip.id), func.sum(Clip.duration))
                         .where(Clip.reviewed_at >= since).group_by(func.date(Clip.reviewed_at), Clip.status)).all()
    days: dict = {}
    for day, status, n, dur in per_day:
        d = days.setdefault(day, {"day": day, "approved": 0, "rejected": 0, "approved_hours": 0.0})
        if status in ("approved", "rejected"):
            d[status] += n
        if status == "approved":
            d["approved_hours"] = round((dur or 0) / 3600, 4)
    agreement = db.execute(select(func.avg(Clip.asr_agreement), func.count(Clip.asr_agreement))).one()
    return {
        "by_status": by_status,
        "per_speaker": per_speaker,
        "per_language": _breakdown(db, Clip.language),
        "needs_review": needs_review,
        "avg_quality": round(avg_quality, 3) if avg_quality is not None else None,
        "edited_transcripts": edited,
        "jobs": {"queued": jobs.get("queued", 0), "running": jobs.get("running", 0)},
        "sources": sources,
        "raw_hours": round(raw_hours / 3600, 3),
        "genders": genders,
        "quality_hist": {"bins": bins, "counts": qhist},
        "flags": dict(flag_counts.most_common()),
        "reject_reasons": reject_reasons,
        "review_per_day": [days[k] for k in sorted(days)],
        "asr_agreement": {"mean_wer": round(agreement[0], 4) if agreement[0] is not None else None,
                          "clips": agreement[1]},
    }
