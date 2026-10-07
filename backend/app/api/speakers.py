"""Speakers."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import case, func, select
from sqlalchemy.orm import Session

from ..db import get_db, get_db_rw
from ..models import Clip, SourceFile, Speaker
from ..schemas import SpeakerIn, SpeakerOut, SpeakerUpdate
from .common import get_or_404

router = APIRouter(tags=["speakers"])


def _with_stats(db: Session, speakers: list[Speaker]) -> list[SpeakerOut]:
    ids = [s.id for s in speakers]
    sources = dict(db.execute(select(SourceFile.speaker_id, func.count(SourceFile.id))
                              .where(SourceFile.speaker_id.in_(ids)).group_by(SourceFile.speaker_id)).all())
    approved = Clip.status == "approved"
    clip_rows = db.execute(
        select(Clip.speaker_id, func.count(Clip.id), func.sum(Clip.duration),
               func.sum(case((approved, 1), else_=0)), func.sum(case((approved, Clip.duration), else_=0)))
        .where(Clip.speaker_id.in_(ids)).group_by(Clip.speaker_id)
    ).all()
    stats = {sid: (n, d or 0, an or 0, ad or 0) for sid, n, d, an, ad in clip_rows}
    out = []
    for s in speakers:
        n, d, an, ad = stats.get(s.id, (0, 0, 0, 0))
        out.append(SpeakerOut(id=s.id, name=s.name, gender=s.gender, notes=s.notes, created_at=s.created_at,
                              sources=sources.get(s.id, 0), clips=n, approved_clips=an,
                              approved_hours=round(ad / 3600, 3), total_hours=round(d / 3600, 3)))
    return out


@router.get("/speakers", response_model=list[SpeakerOut])
def list_speakers(db: Session = Depends(get_db)) -> list[SpeakerOut]:
    return _with_stats(db, list(db.scalars(select(Speaker).order_by(Speaker.name))))


@router.post("/speakers", response_model=SpeakerOut)
def create_speaker(body: SpeakerIn, db: Session = Depends(get_db_rw)) -> SpeakerOut:
    if db.scalars(select(Speaker).where(func.lower(Speaker.name) == body.name.lower())).first():
        raise HTTPException(409, f"speaker '{body.name}' already exists")
    sp = Speaker(name=body.name, gender=body.gender, notes=body.notes)
    db.add(sp)
    db.commit()
    return _with_stats(db, [sp])[0]


@router.patch("/speakers/{speaker_id}", response_model=SpeakerOut)
def update_speaker(speaker_id: int, body: SpeakerUpdate, db: Session = Depends(get_db_rw)) -> SpeakerOut:
    sp = get_or_404(db, Speaker, speaker_id)
    if body.gender is not None:
        sp.gender = body.gender
    if body.notes is not None:
        sp.notes = body.notes or None
    db.commit()
    return _with_stats(db, [sp])[0]
