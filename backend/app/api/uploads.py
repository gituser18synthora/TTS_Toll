"""Upload endpoint.

The request body is the raw file (streamed straight to disk while hashing, so
multi-GB recordings never sit in memory and are written only once); metadata
travels as query parameters:

  curl -X POST --data-binary @talk.mp3 \
    "http://127.0.0.1:8000/api/uploads?filename=talk.mp3&speaker_name=ravi&gender=male&language=hi"

Multi-speaker recordings: pass multi_speaker=true (no speaker_name / gender). The file is
diarized and waits in status needs_speakers until each detected speaker is named.
"""
from __future__ import annotations

import hashlib
import os
import uuid
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from sqlalchemy import func, select

from ..config import Settings
from ..db import session_scope
from ..models import SourceFile, Speaker
from ..paths import DataPaths
from ..pipeline.convert import FFmpegError, probe
from ..schemas import Gender, SourceOut, check_speaker_name
from ..worker.jobs import enqueue
from .common import check_language, get_or_create_speaker, paths_dep, settings_dep, source_out

router = APIRouter(tags=["uploads"])


def _precheck(name: str | None, gender: str | None) -> None:
    if name is None:  # multi-speaker upload: speakers are named after diarization
        return
    with session_scope() as s:
        sp = s.scalars(select(Speaker).where(func.lower(Speaker.name) == name.lower())).first()
        if sp is not None and sp.gender != gender:
            raise HTTPException(409, f"speaker '{sp.name}' already exists with gender '{sp.gender}'")


def _finalize(tmp: Path, sha: str, size: int, filename: str, name: str | None, gender: str | None, language: str,
              notes: str | None, auto_process: bool, multi_speaker: bool, paths: DataPaths,
              settings: Settings) -> SourceOut:
    with session_scope() as s:
        dup = s.scalars(select(SourceFile).where(SourceFile.original_sha256 == sha)).first()
        if dup is not None:
            raise HTTPException(409, f"identical file already uploaded as source #{dup.id} ({dup.original_filename})")
    try:
        info = probe(tmp, settings.audio.ffprobe)
    except FFmpegError as exc:
        raise HTTPException(400, f"not a decodable audio file: {exc}") from exc

    with session_scope(write=True) as s:
        speaker = None if multi_speaker else get_or_create_speaker(s, name, gender)
        folder = speaker.name if speaker else "_multi_speaker"
        dst = paths.unique_raw_path(folder, filename, datetime.now().strftime("%Y%m%d_%H%M%S"))
        os.replace(tmp, dst)
        src = SourceFile(
            original_path=paths.rel(dst), original_filename=filename, original_sha256=sha, original_size=size,
            speaker_id=speaker.id if speaker else None, multi_speaker=multi_speaker, language=language,
            duration=info["duration"], notes=notes or None, status="uploaded", progress=0.0,
        )
        s.add(src)
        s.flush()
        job = enqueue(s, "process", source_file_id=src.id) if auto_process else None
        s.refresh(src)
        return source_out(src, job=job)


@router.post("/uploads", response_model=SourceOut)
async def upload(
    request: Request,
    filename: str,
    language: str,
    speaker_name: str | None = None,
    gender: Gender | None = None,
    notes: str | None = None,
    auto_process: bool = True,
    multi_speaker: bool = False,
    settings: Settings = Depends(settings_dep),
    paths: DataPaths = Depends(paths_dep),
) -> SourceOut:
    name = None
    if not multi_speaker:
        if not speaker_name or not gender:
            raise HTTPException(422, "speaker_name and gender are required (or set multi_speaker=true)")
        try:
            name = check_speaker_name(speaker_name)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
    check_language(settings, language)
    filename = Path(filename).name.strip() or "upload"
    ext = Path(filename).suffix.lower().lstrip(".")
    if ext not in settings.upload.allowed_extensions:
        raise HTTPException(415, f"unsupported file type '.{ext}'; allowed: {settings.upload.allowed_extensions}")
    await run_in_threadpool(_precheck, name, gender)

    paths.tmp.mkdir(parents=True, exist_ok=True)
    tmp = paths.tmp / f"upload_{uuid.uuid4().hex}.part"
    limit = int(settings.upload.max_file_gb * 1024**3)
    digest, size = hashlib.sha256(), 0
    try:
        with open(tmp, "wb") as fh:
            async for chunk in request.stream():
                size += len(chunk)
                if size > limit:
                    raise HTTPException(413, f"file larger than {settings.upload.max_file_gb} GB")
                digest.update(chunk)
                fh.write(chunk)
        if size == 0:
            raise HTTPException(400, "empty upload")
        return await run_in_threadpool(_finalize, tmp, digest.hexdigest(), size, filename, name,
                                       None if multi_speaker else gender, language, notes, auto_process,
                                       multi_speaker, paths, settings)
    finally:
        tmp.unlink(missing_ok=True)  # already moved into data/raw on success
