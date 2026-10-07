"""Whole-database jobs: re-check speaker consistency, find duplicates."""
from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..db import get_db_rw
from ..models import Job
from ..schemas import JobOut
from ..worker.jobs import enqueue

router = APIRouter(tags=["maintenance"])


def _enqueue_once(db: Session, job_type: str) -> JobOut:
    job = db.scalars(select(Job).where(Job.type == job_type, Job.status.in_(("queued", "running")))).first()
    if job is None:
        job = enqueue(db, job_type)
        db.commit()
    return JobOut.model_validate(job)


@router.post("/maintenance/speaker-check", response_model=JobOut)
def speaker_check(db: Session = Depends(get_db_rw)) -> JobOut:
    """Re-check every speaker's clips against their voice centroid (after many new uploads)."""
    return _enqueue_once(db, "speaker_check")


@router.post("/maintenance/dedup", response_model=JobOut)
def dedup_all(db: Session = Depends(get_db_rw)) -> JobOut:
    """Look for exact / near-identical duplicate clips across all files."""
    return _enqueue_once(db, "dedup")
