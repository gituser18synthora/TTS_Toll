"""DB-backed job queue (no Redis/Celery): enqueue, claim, heartbeat, cancel, log."""
from __future__ import annotations

import json
import logging
import os
import socket
import time
from collections.abc import Callable
from datetime import timedelta

from sqlalchemy import select, text, update
from sqlalchemy.orm import Session

from ..db import session_scope
from ..models import Job, SourceFile, utcnow

log = logging.getLogger(__name__)

ACTIVE = ("queued", "running")
MAX_LOG_CHARS = 200_000


class JobCancelled(Exception):
    pass


def active_job_for_source(session: Session, source_id: int) -> Job | None:
    return session.scalars(
        select(Job).where(Job.source_file_id == source_id, Job.status.in_(ACTIVE)).order_by(Job.id.desc())
    ).first()


def enqueue(session: Session, job_type: str, *, source_file_id: int | None = None,
            export_id: int | None = None, params: dict | None = None) -> Job:
    """Create a queued job. For source jobs an already-active job is returned instead of a duplicate."""
    if source_file_id is not None:
        existing = active_job_for_source(session, source_file_id)
        if existing is not None:
            return existing
    job = Job(type=job_type, source_file_id=source_file_id, export_id=export_id,
              params=json.dumps(params or {}), status="queued", message="queued")
    session.add(job)
    session.flush()
    return job


def claim_next(worker_id: str) -> int | None:
    """Atomically move the oldest queued job to running. Returns its id."""
    with session_scope(write=True) as s:
        row = s.execute(
            text(
                "UPDATE jobs SET status='running', worker_id=:w, started_at=:now, heartbeat_at=:now, "
                "message='starting' "
                "WHERE id = (SELECT id FROM jobs WHERE status='queued' ORDER BY id LIMIT 1) "
                "AND status='queued' RETURNING id"
            ),
            {"w": worker_id, "now": utcnow()},
        ).first()
        return int(row[0]) if row else None


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def worker_is_dead(worker_id: str | None) -> bool:
    """True if worker_id ("host:pid") ran on this machine and that process is gone."""
    if not worker_id or ":" not in worker_id:
        return False
    host, _, pid = worker_id.rpartition(":")
    return host == socket.gethostname() and pid.isdigit() and not _pid_alive(int(pid))


def requeue_stale(stale_sec: float) -> list[int]:
    """Running jobs whose worker died (or stopped heart-beating) go back to the queue; they resume."""
    cutoff = utcnow() - timedelta(seconds=stale_sec)
    with session_scope(write=True) as s:
        jobs = [j for j in s.scalars(select(Job).where(Job.status == "running"))
                if (j.heartbeat_at is None or j.heartbeat_at < cutoff) or worker_is_dead(j.worker_id)]
        for job in jobs:
            job.status = "queued"
            job.message = "re-queued after worker restart"
            job.log = (job.log or "") + f"[{utcnow():%H:%M:%S}] worker lost; job re-queued (will resume)\n"
        return [j.id for j in jobs]


def requeue_own(worker_id: str) -> None:
    """On graceful shutdown put the in-flight job back in the queue."""
    with session_scope(write=True) as s:
        for job in s.scalars(select(Job).where(Job.status == "running", Job.worker_id == worker_id)):
            job.status = "queued"
            job.message = "interrupted; will resume"
            job.log = (job.log or "") + f"[{utcnow():%H:%M:%S}] worker stopped; job re-queued (will resume)\n"


class JobContext:
    """Handed to pipeline code: progress, logging, cancellation."""

    def __init__(self, job_id: int, source_file_id: int | None = None):
        self.job_id = job_id
        self.source_file_id = source_file_id
        self._last_progress_write = 0.0
        self._last_cancel_check = 0.0
        self._progress = 0.0
        self._message: str | None = None
        self.logger = logging.getLogger(f"job.{job_id}")

    # -- logging -------------------------------------------------------------
    def log(self, msg: str, level: int = logging.INFO) -> None:
        self.logger.log(level, msg)
        line = f"[{utcnow():%H:%M:%S}] {logging.getLevelName(level) + ': ' if level >= logging.WARNING else ''}{msg}\n"
        with session_scope(write=True) as s:
            job = s.get(Job, self.job_id)
            if job is not None:
                job.log = ((job.log or "") + line)[-MAX_LOG_CHARS:]
                job.heartbeat_at = utcnow()

    def warn(self, msg: str) -> None:
        self.log(msg, logging.WARNING)

    # -- progress ------------------------------------------------------------
    def progress(self, fraction: float, message: str | None = None, force: bool = False) -> None:
        self._progress = max(0.0, min(1.0, fraction))
        if message is not None:
            self._message = message
        now = time.monotonic()
        if not force and now - self._last_progress_write < 1.0:
            return
        self._last_progress_write = now
        pct = round(self._progress * 100, 1)
        with session_scope(write=True) as s:
            s.execute(update(Job).where(Job.id == self.job_id)
                      .values(progress=pct, message=(self._message or "")[:250], heartbeat_at=utcnow()))
            if self.source_file_id is not None:
                s.execute(update(SourceFile).where(SourceFile.id == self.source_file_id).values(progress=pct))

    def sub(self, start: float, weight: float, message: str) -> Callable[..., None]:
        """Progress callback mapping 0..1 of a step onto [start, start+weight] of the job.

        The callback also accepts an optional detail string shown next to the step name.
        """
        self.progress(start, message, force=True)

        def report(frac: float, detail: str | None = None) -> None:
            self.progress(start + weight * max(0.0, min(1.0, frac)),
                          f"{message}: {detail}" if detail else None)

        return report

    # -- cancellation ---------------------------------------------------------
    def check_cancel(self) -> None:
        now = time.monotonic()
        if now - self._last_cancel_check < 1.0:
            return
        self._last_cancel_check = now
        with session_scope() as s:
            flag = s.scalar(select(Job.cancel_requested).where(Job.id == self.job_id))
        if flag:
            raise JobCancelled()
