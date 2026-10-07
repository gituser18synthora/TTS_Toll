"""Source recordings: list, edit metadata, (re)process, stream audio."""
from __future__ import annotations

import json
import re
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from sqlalchemy import select, update
from sqlalchemy.orm import Session

from ..config import Settings
from ..db import get_db, get_db_rw
from ..models import Clip, SourceFile, Speaker
from ..paths import DataPaths
from ..schemas import SourceOut, SourceUpdate, SpeakerAssignments
from ..worker.jobs import active_job_for_source, enqueue
from .common import (check_language, clip_counts, get_or_404, get_or_create_speaker, latest_jobs, paths_dep,
                     settings_dep, source_out)

router = APIRouter(tags=["sources"])


@router.get("/sources", response_model=list[SourceOut])
def list_sources(speaker_id: int | None = None, language: str | None = None, status: str | None = None,
                 db: Session = Depends(get_db)) -> list[SourceOut]:
    q = select(SourceFile).order_by(SourceFile.id.desc())
    if speaker_id:
        q = q.where(SourceFile.speaker_id == speaker_id)
    if language:
        q = q.where(SourceFile.language == language)
    if status:
        q = q.where(SourceFile.status == status)
    sources = db.scalars(q).unique().all()
    ids = [s.id for s in sources]
    counts = clip_counts(db, ids)
    jobs = latest_jobs(db, source_ids=ids)
    return [source_out(s, counts.get(s.id), jobs.get(s.id)) for s in sources]


@router.get("/sources/{source_id}", response_model=SourceOut)
def get_source(source_id: int, db: Session = Depends(get_db)) -> SourceOut:
    s = get_or_404(db, SourceFile, source_id)
    return source_out(s, clip_counts(db, [s.id]).get(s.id), latest_jobs(db, source_ids=[s.id]).get(s.id))


@router.patch("/sources/{source_id}", response_model=SourceOut)
def update_source(source_id: int, body: SourceUpdate, db: Session = Depends(get_db_rw),
                  settings: Settings = Depends(settings_dep)) -> SourceOut:
    """Fix upload metadata; speaker/language changes propagate to the source's clips."""
    s = get_or_404(db, SourceFile, source_id)
    clip_values = {}
    if body.speaker_name is not None and s.multi_speaker:
        raise HTTPException(409, "multi-speaker file: change its speakers with Assign speakers")
    if body.speaker_name is not None:
        gender = body.gender or s.speaker.gender
        speaker = get_or_create_speaker(db, body.speaker_name, gender)
        s.speaker_id = speaker.id
        clip_values["speaker_id"] = speaker.id
    if body.language is not None:
        s.language = check_language(settings, body.language)
        clip_values["language"] = body.language
    if body.notes is not None:
        s.notes = body.notes or None
    if clip_values:
        db.execute(update(Clip).where(Clip.source_file_id == s.id).values(**clip_values))
    db.commit()
    db.refresh(s)
    return get_source(source_id, db)


def _enqueue_process(db: Session, source_id: int, reset: bool) -> SourceOut:
    s = get_or_404(db, SourceFile, source_id)
    if active_job_for_source(db, s.id) is not None:
        raise HTTPException(409, "a job for this file is already queued or running")
    enqueue(db, "process", source_file_id=s.id, params={"reset": True} if reset else None)
    s.error = None
    if s.status == "failed":
        s.status = "uploaded"
    db.commit()
    return get_source(source_id, db)


@router.post("/sources/{source_id}/process", response_model=SourceOut)
def process_source(source_id: int, db: Session = Depends(get_db_rw)) -> SourceOut:
    """Start / retry / resume: continues from the last completed step."""
    return _enqueue_process(db, source_id, reset=False)


@router.post("/sources/{source_id}/reprocess", response_model=SourceOut)
def reprocess_source(source_id: int, db: Session = Depends(get_db_rw)) -> SourceOut:
    """Discard all clips and derived audio of this file and run the pipeline from scratch."""
    return _enqueue_process(db, source_id, reset=True)


@router.post("/sources/{source_id}/retranscribe", response_model=SourceOut)
def retranscribe_source(source_id: int, db: Session = Depends(get_db_rw)) -> SourceOut:
    """Re-run both ASRs (+ agreement) on the existing clips without re-segmenting.

    Approved / rejected clips and edited transcripts are kept; pending, never-edited
    transcripts take the new primary text. Use after switching to a better ASR model.
    """
    s = get_or_404(db, SourceFile, source_id)
    if "clips" not in s.get_steps_done():
        raise HTTPException(409, "this file has no clips yet; process it first")
    if active_job_for_source(db, s.id) is not None:
        raise HTTPException(409, "a job for this file is already queued or running")
    enqueue(db, "retranscribe", source_file_id=s.id)
    db.commit()
    return get_source(source_id, db)


@router.post("/sources/retranscribe-all")
def retranscribe_all(db: Session = Depends(get_db_rw)) -> dict:
    """Queue a re-transcribe job for every file that has clips and no active job."""
    queued, skipped = [], []
    for s in db.scalars(select(SourceFile).order_by(SourceFile.id)).unique():
        if "clips" in s.get_steps_done() and active_job_for_source(db, s.id) is None:
            enqueue(db, "retranscribe", source_file_id=s.id)
            queued.append(s.id)
        else:
            skipped.append(s.id)
    db.commit()
    return {"queued": queued, "skipped": skipped}


@router.get("/sources/{source_id}/audio")
def source_audio(source_id: int, kind: Literal["original", "converted", "enhanced"] = "converted",
                 db: Session = Depends(get_db), paths: DataPaths = Depends(paths_dep)) -> FileResponse:
    s = get_or_404(db, SourceFile, source_id)
    rel = {"original": s.original_path, "converted": s.converted_path, "enhanced": s.enhanced_path}[kind]
    if not rel or not paths.abs(rel).exists():
        raise HTTPException(404, f"no {kind} audio for source {source_id}")
    return FileResponse(paths.abs(rel), filename=f"{kind}_{s.original_filename}" if kind == "original" else None)


# ---------------------------------------------------------------- multi-speaker files
def _diarization(paths: DataPaths, source_id: int) -> dict:
    path = paths.work_dir(source_id) / "diarization.json"
    if not path.exists():
        raise HTTPException(404, "no diarization for this file (not multi-speaker, or not diarized yet)")
    return json.loads(path.read_text(encoding="utf-8"))


def _speakers_view(db: Session, src: SourceFile, paths: DataPaths) -> dict:
    diar = _diarization(paths, src.id)
    mapping = src.get_speaker_map()
    labels = []
    for label, info in sorted(diar["labels"].items()):
        m = mapping.get(label, {})
        sp = db.get(Speaker, m["speaker_id"]) if m.get("speaker_id") else None
        labels.append({
            "label": label, "seconds": info["seconds"], "turns": info["turns"],
            "samples": [f"/api/sources/{src.id}/speaker-samples/{p.rsplit('/', 1)[-1]}" for p in info["samples"]],
            "ignore": bool(m.get("ignore")), "speaker_name": sp.name if sp else None,
            "gender": sp.gender if sp else None,
        })
    return {"source_id": src.id, "engine": diar.get("engine"), "status": src.status,
            "overlap_seconds": round(sum(b - a for a, b in diar.get("overlaps", [])), 1),
            "clips_created": "clips" in src.get_steps_done(), "labels": labels}


@router.get("/sources/{source_id}/speakers")
def get_source_speakers(source_id: int, db: Session = Depends(get_db), paths: DataPaths = Depends(paths_dep)) -> dict:
    """Speakers found by diarization, with sample clips to listen to and current names."""
    return _speakers_view(db, get_or_404(db, SourceFile, source_id), paths)


@router.get("/sources/{source_id}/speaker-samples/{name}")
def speaker_sample(source_id: int, name: str, paths: DataPaths = Depends(paths_dep)) -> FileResponse:
    if not re.fullmatch(r"[A-Za-z0-9_]{1,40}\.wav", name):
        raise HTTPException(400, "bad sample name")
    path = paths.work_dir(source_id) / "speaker_samples" / name
    if not path.exists():
        raise HTTPException(404, "sample not found")
    return FileResponse(path, media_type="audio/wav")


@router.post("/sources/{source_id}/speakers")
def assign_speakers(source_id: int, body: SpeakerAssignments, db: Session = Depends(get_db_rw),
                    paths: DataPaths = Depends(paths_dep), settings: Settings = Depends(settings_dep)) -> dict:
    """Name (or ignore) each diarized speaker. Before clips exist this resumes processing; afterwards
    renaming re-labels the existing clips (switching ignore on/off needs Reprocess)."""
    src = get_or_404(db, SourceFile, source_id)
    if not src.multi_speaker:
        raise HTTPException(409, "not a multi-speaker file")
    known = set(_diarization(paths, src.id)["labels"])
    given = {a.label for a in body.assignments}
    if given - known:
        raise HTTPException(422, f"unknown speaker label(s): {sorted(given - known)}")
    mapping = src.get_speaker_map()
    new_map = dict(mapping)
    for a in body.assignments:
        if a.ignore:
            new_map[a.label] = {"ignore": True}
            continue
        if not a.name or not a.gender:
            raise HTTPException(422, f"{a.label}: name and gender are required unless ignored")
        new_map[a.label] = {"speaker_id": get_or_create_speaker(db, a.name, a.gender).id}
    if not any(not m.get("ignore") for m in new_map.values()) and known <= set(new_map):
        raise HTTPException(422, "every speaker is ignored: nothing would be kept")

    clips_exist = "clips" in src.get_steps_done()
    if clips_exist:
        changed_ignore = {lab for lab in new_map if bool(new_map[lab].get("ignore")) != bool(mapping.get(lab, {}).get("ignore"))}
        if changed_ignore:
            raise HTTPException(409, f"clips already exist; changing which speakers are ignored ({sorted(changed_ignore)}) "
                                     f"needs Reprocess")
        for label, m in new_map.items():
            if not m.get("ignore"):
                db.execute(update(Clip).where(Clip.source_file_id == src.id, Clip.speaker_label == label)
                           .values(speaker_id=m["speaker_id"]))
        src.speaker_map = json.dumps(new_map)
        if settings.speaker_check.enabled and active_job_for_source(db, src.id) is None:
            enqueue(db, "speaker_check")  # centroids changed for the speakers involved
    else:
        src.speaker_map = json.dumps(new_map)
        if known <= set(new_map) and active_job_for_source(db, src.id) is None:
            enqueue(db, "process", source_file_id=src.id)  # resume: assign -> clips -> ...
            src.status, src.error = "uploaded", None
    db.commit()
    return _speakers_view(db, src, paths)
