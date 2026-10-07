"""Dataset exports (approved clips only)."""
from __future__ import annotations

import json

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..config import Settings
from ..db import get_db, get_db_rw
from ..export.exporter import preview
from ..models import Export
from ..paths import DataPaths, slugify
from ..schemas import ExportCreate, ExportOut
from ..worker.jobs import enqueue
from .common import check_language, export_out, get_or_404, latest_jobs, paths_dep, settings_dep

router = APIRouter(tags=["exports"])


@router.get("/exports/preview")
def export_preview(speaker_ids: list[int] = Query(default=[]), languages: list[str] = Query(default=[]),
                   db: Session = Depends(get_db)) -> dict:
    return preview(db, speaker_ids or None, languages or None)


@router.post("/exports", response_model=ExportOut)
def create_export(body: ExportCreate, db: Session = Depends(get_db_rw), settings: Settings = Depends(settings_dep),
                  paths: DataPaths = Depends(paths_dep)) -> ExportOut:
    for lang in body.languages:
        check_language(settings, lang)
    params = body.model_dump(exclude={"name"})
    exp = Export(name=slugify(body.name, 64), params=json.dumps(params), status="queued")
    db.add(exp)
    db.flush()
    job = enqueue(db, "export", export_id=exp.id)
    db.commit()
    return export_out(exp, paths, job)


@router.get("/exports", response_model=list[ExportOut])
def list_exports(db: Session = Depends(get_db), paths: DataPaths = Depends(paths_dep)) -> list[ExportOut]:
    exports = db.scalars(select(Export).order_by(Export.id.desc()).limit(200)).all()
    jobs = latest_jobs(db, export_ids=[e.id for e in exports])
    return [export_out(e, paths, jobs.get(e.id)) for e in exports]


@router.get("/exports/{export_id}", response_model=ExportOut)
def get_export(export_id: int, db: Session = Depends(get_db), paths: DataPaths = Depends(paths_dep)) -> ExportOut:
    e = get_or_404(db, Export, export_id)
    return export_out(e, paths, latest_jobs(db, export_ids=[e.id]).get(e.id))


@router.post("/exports/{export_id}/rerun", response_model=ExportOut)
def rerun_export(export_id: int, db: Session = Depends(get_db_rw), paths: DataPaths = Depends(paths_dep)) -> ExportOut:
    """Rebuild the export from the *current* approved clips (same name and options)."""
    e = get_or_404(db, Export, export_id)
    if e.status in ("queued", "running"):
        raise HTTPException(409, "export is already running")
    e.status, e.error = "queued", None
    job = enqueue(db, "export", export_id=e.id)
    db.commit()
    return export_out(e, paths, job)


@router.get("/exports/{export_id}/download")
def download_export(export_id: int, db: Session = Depends(get_db),
                    paths: DataPaths = Depends(paths_dep)) -> FileResponse:
    e = get_or_404(db, Export, export_id)
    if not e.zip_path or not paths.abs(e.zip_path).exists():
        raise HTTPException(404, "no zip for this export")
    return FileResponse(paths.abs(e.zip_path), media_type="application/zip", filename=f"{e.name}.zip")
