"""Background jobs: list, inspect logs, cancel, retry."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..db import get_db, get_db_rw
from ..models import Export, Job, SourceFile, utcnow
from ..schemas import JobDetail, JobOut
from ..worker.jobs import active_job_for_source, enqueue
from .common import get_or_404

router = APIRouter(tags=["jobs"])


@router.get("/jobs", response_model=list[JobOut])
def list_jobs(status: str | None = None, type: str | None = None, source_file_id: int | None = None,
              limit: int = Query(default=50, ge=1, le=500), db: Session = Depends(get_db)) -> list[JobOut]:
    q = select(Job).order_by(Job.id.desc()).limit(limit)
    if status:
        q = q.where(Job.status.in_(status.split(",")))
    if type:
        q = q.where(Job.type == type)
    if source_file_id:
        q = q.where(Job.source_file_id == source_file_id)
    return [JobOut.model_validate(j) for j in db.scalars(q)]


@router.get("/jobs/{job_id}", response_model=JobDetail)
def get_job(job_id: int, db: Session = Depends(get_db)) -> JobDetail:
    job = get_or_404(db, Job, job_id)
    return JobDetail.model_validate(job)


@router.post("/jobs/{job_id}/cancel", response_model=JobOut)
def cancel_job(job_id: int, db: Session = Depends(get_db_rw)) -> JobOut:
    job = get_or_404(db, Job, job_id)
    if job.status == "queued":
        job.status, job.message, job.finished_at = "cancelled", "cancelled", utcnow()
        if job.source_file_id:
            src = db.get(SourceFile, job.source_file_id)
            if src.status != "done":
                src.status, src.error = "failed", "Cancelled by user"
        if job.export_id:
            exp = db.get(Export, job.export_id)
            exp.status, exp.error = "failed", "Cancelled by user"
    elif job.status == "running":
        job.cancel_requested = True
        job.message = "cancelling…"
    else:
        raise HTTPException(409, f"job is already {job.status}")
    db.commit()
    return JobOut.model_validate(job)


@router.post("/jobs/{job_id}/retry", response_model=JobOut)
def retry_job(job_id: int, db: Session = Depends(get_db_rw)) -> JobOut:
    """Re-run a failed/cancelled job. Processing jobs resume from the last completed step."""
    job = get_or_404(db, Job, job_id)
    if job.status in ("queued", "running"):
        raise HTTPException(409, "job is still active")
    if job.type == "process":
        if active_job_for_source(db, job.source_file_id) is not None:
            raise HTTPException(409, "a job for this file is already queued or running")
        new = enqueue(db, "process", source_file_id=job.source_file_id)
        src = db.get(SourceFile, job.source_file_id)
        src.error = None
        if src.status == "failed":
            src.status = "uploaded"
    elif job.type == "retranscribe":
        if active_job_for_source(db, job.source_file_id) is not None:
            raise HTTPException(409, "a job for this file is already queued or running")
        # continue where it stopped: clips that already got new transcripts are not redone
        new = enqueue(db, "retranscribe", source_file_id=job.source_file_id,
                      params={"cleared": bool(job.get_params().get("cleared"))})
        src = db.get(SourceFile, job.source_file_id)
        src.error = None
    elif job.type in ("speaker_check", "dedup"):
        new = enqueue(db, job.type)
    elif job.type == "export":
        new = enqueue(db, "export", export_id=job.export_id)
        exp = db.get(Export, job.export_id)
        exp.status, exp.error = "queued", None
    else:
        raise HTTPException(400, f"cannot retry job type {job.type}")
    db.commit()
    return JobOut.model_validate(new)
