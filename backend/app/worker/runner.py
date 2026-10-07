"""Worker loop: claims queued jobs from the DB and runs them one at a time.

Run standalone with `python -m app.worker`, or embedded in the API process
(worker.embedded: true). Ctrl+C / SIGTERM puts the in-flight job back in the
queue; it resumes from its last completed step on the next start.
"""
from __future__ import annotations

import json
import logging
import os
import signal
import socket
import threading
import time
import traceback

from sqlalchemy import select, update

from .. import engines
from ..config import Settings, configure_model_cache
from ..db import create_schema, init_engine, session_scope
from ..devices import resolve_device
from ..logging_setup import setup_logging
from ..models import Export, Job, SourceFile, WorkerInfo, utcnow
from ..paths import DataPaths
from .jobs import JobCancelled, JobContext, claim_next, requeue_own, requeue_stale, worker_is_dead

log = logging.getLogger(__name__)


def _finish(job_id: int, status: str, error: str | None = None) -> None:
    with session_scope(write=True) as s:
        job = s.get(Job, job_id)
        job.status = status
        job.finished_at = utcnow()
        job.error = error
        if status == "done":
            job.progress = 100.0
        job.message = {"done": "done", "failed": "failed", "cancelled": "cancelled"}.get(status, status)


def _fail_target(job: Job, message: str) -> None:
    with session_scope(write=True) as s:
        if job.source_file_id:
            s.execute(update(SourceFile).where(SourceFile.id == job.source_file_id)
                      .values(status="failed", error=message[:2000]))
        if job.export_id:
            s.execute(update(Export).where(Export.id == job.export_id)
                      .values(status="failed", error=message[:2000], finished_at=utcnow()))


def run_job(job_id: int, settings: Settings) -> None:
    from ..export.exporter import run_export
    from ..pipeline.orchestrator import SourcePipeline, run_dedup_all, run_speaker_check

    with session_scope() as s:
        job = s.get(Job, job_id)
    ctx = JobContext(job.id, job.source_file_id)
    params = job.get_params()
    ctx.log(f"job {job.id} ({job.type}) started" + (f" with {params}" if params else ""))
    try:
        if job.type == "process":
            SourcePipeline(ctx, settings, job.source_file_id).run(reset=bool(params.get("reset")))
        elif job.type == "retranscribe":
            SourcePipeline(ctx, settings, job.source_file_id).retranscribe(params)
        elif job.type == "export":
            run_export(ctx, settings, job.export_id)
        elif job.type == "speaker_check":
            run_speaker_check(ctx, settings)
        elif job.type == "dedup":
            run_dedup_all(ctx, settings)
        else:
            raise ValueError(f"unknown job type {job.type!r}")
        _finish(job.id, "done")
        ctx.log("job finished")
    except JobCancelled:
        ctx.warn("cancelled by user")
        _finish(job.id, "cancelled")
        _fail_target(job, "Cancelled by user")
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as exc:  # noqa: BLE001 - every failure must reach the UI
        tb = traceback.format_exc()
        log.error("job %s failed:\n%s", job.id, tb)
        ctx.log(f"FAILED: {exc}\n{tb}", logging.ERROR)
        _finish(job.id, "failed", f"{type(exc).__name__}: {exc}")
        _fail_target(job, f"{type(exc).__name__}: {exc}")


class Worker:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.worker_id = f"{socket.gethostname()}:{os.getpid()}"
        self.stop_event = threading.Event()
        self.device = "unknown"
        self._last_beat = 0.0
        self._last_stale_check = 0.0

    def _beat(self, force: bool = False) -> None:
        now = time.monotonic()
        if not force and now - self._last_beat < 10:
            return
        self._last_beat = now
        with session_scope(write=True) as s:
            info = s.get(WorkerInfo, self.worker_id)
            if info is None:
                info = WorkerInfo(id=self.worker_id, device=self.device)
                s.add(info)
            info.info = json.dumps({"pid": os.getpid(), "engines": engines.snapshot()})
            info.heartbeat_at = utcnow()
            for other in s.scalars(select(WorkerInfo).where(WorkerInfo.id != self.worker_id)):
                if worker_is_dead(other.id):
                    s.delete(other)

    def _unregister(self) -> None:
        try:
            with session_scope(write=True) as s:
                info = s.get(WorkerInfo, self.worker_id)
                if info is not None:
                    s.delete(info)
        except Exception:  # pragma: no cover - best effort on shutdown
            pass

    def _heartbeat_loop(self) -> None:
        # separate thread: a long job must not make the worker look offline
        while not self.stop_event.wait(10):
            try:
                self._beat(force=True)
            except Exception:  # noqa: BLE001 - transient DB contention; try again next tick
                log.debug("worker heartbeat failed", exc_info=True)

    def run(self) -> None:
        self.device = resolve_device(self.settings.device)
        log.info("worker %s started (device=%s)", self.worker_id, self.device)
        self._beat(force=True)
        threading.Thread(target=self._heartbeat_loop, name="worker-heartbeat", daemon=True).start()
        requeued = requeue_stale(self.settings.worker.stale_job_sec)
        if requeued:
            log.info("re-queued stale jobs: %s", requeued)
        try:
            while not self.stop_event.is_set():
                job_id = claim_next(self.worker_id)
                if job_id is None:
                    self.stop_event.wait(self.settings.worker.poll_interval_sec)
                    if time.monotonic() - self._last_stale_check > 60:
                        self._last_stale_check = time.monotonic()
                        requeue_stale(self.settings.worker.stale_job_sec)
                    continue
                log.info("claimed job %s", job_id)
                run_job(job_id, self.settings)
        except (KeyboardInterrupt, SystemExit):
            log.info("worker stopping; re-queueing in-flight job")
            requeue_own(self.worker_id)
        finally:
            self.stop_event.set()
            self._unregister()


def start_embedded(settings: Settings) -> Worker:
    worker = Worker(settings)
    threading.Thread(target=worker.run, name="portal-worker", daemon=True).start()
    return worker


def main() -> None:
    from ..config import get_settings

    settings = get_settings()
    paths = DataPaths.from_settings(settings)
    paths.ensure()
    setup_logging("worker", paths.logs)
    configure_model_cache(settings)
    init_engine(paths.db_file)
    create_schema()

    def _sigterm(_signo, _frame):
        raise SystemExit(0)

    signal.signal(signal.SIGTERM, _sigterm)
    Worker(settings).run()
