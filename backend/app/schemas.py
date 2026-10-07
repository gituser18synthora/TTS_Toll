"""Pydantic request/response models."""
from __future__ import annotations

import json
from datetime import datetime
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, PlainSerializer, field_validator

UtcDatetime = Annotated[
    datetime,
    PlainSerializer(lambda d: d.isoformat() + "Z" if d.tzinfo is None else d.isoformat(), return_type=str),
]
Gender = Literal["male", "female"]
ClipStatus = Literal["pending", "approved", "rejected"]


def check_speaker_name(value: str) -> str:
    """Speaker names become the "<name>: <text>" prefix in training data."""
    name = " ".join(value.split())
    if not 1 <= len(name) <= 64:
        raise ValueError("speaker name must be 1-64 characters")
    if any(ch in name for ch in ":|"):
        raise ValueError("speaker name must not contain ':' or '|'")
    if not any(ch.isalnum() for ch in name):
        raise ValueError("speaker name needs at least one letter or digit")
    return name


class _Out(BaseModel):
    model_config = ConfigDict(from_attributes=True)


class SpeakerIn(BaseModel):
    name: str
    gender: Gender
    notes: str | None = None

    @field_validator("name")
    @classmethod
    def _name(cls, v: str) -> str:
        return check_speaker_name(v)


class SpeakerUpdate(BaseModel):
    gender: Gender | None = None
    notes: str | None = None


class SpeakerOut(_Out):
    id: int
    name: str
    gender: str
    notes: str | None = None
    created_at: UtcDatetime
    sources: int = 0
    clips: int = 0
    approved_clips: int = 0
    approved_hours: float = 0.0
    total_hours: float = 0.0


class JobOut(_Out):
    id: int
    type: str
    source_file_id: int | None = None
    export_id: int | None = None
    status: str
    progress: float
    message: str | None = None
    error: str | None = None
    created_at: UtcDatetime
    started_at: UtcDatetime | None = None
    finished_at: UtcDatetime | None = None


class JobDetail(JobOut):
    log: str = ""
    params: dict = {}

    @field_validator("params", mode="before")
    @classmethod
    def _parse_params(cls, v):
        if isinstance(v, str):
            try:
                return json.loads(v or "{}")
            except ValueError:
                return {}
        return v or {}


class ClipCounts(BaseModel):
    total: int = 0
    pending: int = 0
    approved: int = 0
    rejected: int = 0
    needs_review: int = 0
    approved_hours: float = 0.0
    total_hours: float = 0.0


class SourceOut(_Out):
    id: int
    original_filename: str
    original_path: str
    speaker_id: int | None = None
    speaker_name: str
    gender: str | None = None
    multi_speaker: bool = False
    language: str
    duration: float | None = None
    status: str
    stage: str | None = None
    progress: float
    error: str | None = None
    notes: str | None = None
    steps_done: list[str] = []
    enhanced: bool = False
    created_at: UtcDatetime
    updated_at: UtcDatetime
    counts: ClipCounts = ClipCounts()
    job: JobOut | None = None


class SourceUpdate(BaseModel):
    speaker_name: str | None = None
    gender: Gender | None = None
    language: str | None = None
    notes: str | None = None

    @field_validator("speaker_name")
    @classmethod
    def _name(cls, v: str | None) -> str | None:
        return None if v is None else check_speaker_name(v)


class ClipOut(_Out):
    id: int
    source_file_id: int
    source_filename: str
    denoised: bool = False  # an un-denoised original exists for A/B listening
    speaker_id: int
    speaker_name: str
    gender: str
    language: str
    idx: int
    start_time: float
    end_time: float
    duration: float
    text: str | None = None
    text_edited: bool
    asr_primary_text: str | None = None
    asr_secondary_text: str | None = None
    primary_normalized: str | None = None    # what "use primary" / "use secondary" put in the box
    secondary_normalized: str | None = None
    asr_agreement: float | None = None
    asr_confidence: float | None = None
    asr_primary_engine: str | None = None
    asr_secondary_engine: str | None = None
    quality_score: float | None = None
    quality_info: dict | None = None
    loudness_lufs: float | None = None
    needs_review: bool
    flags: list[str] = []
    speaker_label: str | None = None
    speaker_similarity: float | None = None
    duplicate_of: int | None = None
    status: str
    reject_reason: str | None = None
    reviewed_at: UtcDatetime | None = None
    updated_at: UtcDatetime
    audio_url: str


class ClipPage(BaseModel):
    items: list[ClipOut]
    total: int
    page: int
    page_size: int
    offset: int = 0


class ClipUpdate(BaseModel):
    text: str | None = Field(default=None, max_length=5000)
    status: ClipStatus | None = None
    reject_reason: str | None = Field(default=None, max_length=64)
    needs_review: bool | None = None


class BulkAction(BaseModel):
    ids: list[int] = Field(min_length=1, max_length=2000)
    action: Literal["approve", "reject", "pending"]
    reject_reason: str | None = Field(default=None, max_length=64)


class BulkResult(BaseModel):
    updated: int
    skipped: int
    skipped_reason: str | None = None


class ExportCreate(BaseModel):
    name: str = Field(min_length=1, max_length=64, pattern=r"^[\w\-. ]+$")
    speaker_ids: list[int] = []
    languages: list[str] = []
    formats: list[Literal["jsonl", "csv", "hf"]] = ["jsonl", "csv"]
    # Hindi / Hinglish rows: stored Devanagari, romanized, or one row of each
    script: Literal["devanagari", "roman", "both"] = "devanagari"
    verify_audio: bool = True
    make_zip: bool = True


class ExportOut(_Out):
    id: int
    name: str
    params: dict
    status: str
    out_dir: str | None = None
    zip_path: str | None = None
    zip_size: int | None = None
    clip_count: int
    total_duration: float
    stats: dict | None = None
    error: str | None = None
    created_at: UtcDatetime
    finished_at: UtcDatetime | None = None
    job: JobOut | None = None


class SpeakerAssignment(BaseModel):
    label: str = Field(pattern=r"^[A-Za-z0-9_]{1,32}$")
    ignore: bool = False
    name: str | None = None
    gender: Gender | None = None

    @field_validator("name")
    @classmethod
    def _name(cls, v: str | None) -> str | None:
        return None if v is None else check_speaker_name(v)


class SpeakerAssignments(BaseModel):
    assignments: list[SpeakerAssignment] = Field(min_length=1, max_length=50)
