"""DATA_ROOT layout. The DB only ever stores paths relative to DATA_ROOT."""
from __future__ import annotations

import re
import shutil
from pathlib import Path

from .config import Settings


def source_tag(source_id: int) -> str:
    return f"src_{source_id:05d}"


def slugify(value: str, max_len: int = 48) -> str:
    """Filesystem-safe ASCII-ish slug; keeps unicode letters (Indic names) intact."""
    value = re.sub(r"[^\w\-]+", "_", value.strip(), flags=re.UNICODE)
    value = re.sub(r"_+", "_", value).strip("_")
    return (value or "x")[:max_len]


class DataPaths:
    def __init__(self, root: Path):
        self.root = root.resolve()

    @classmethod
    def from_settings(cls, settings: Settings) -> "DataPaths":
        return cls(settings.data_root)

    # top-level folders -----------------------------------------------------
    @property
    def raw(self) -> Path:
        return self.root / "raw"

    @property
    def converted(self) -> Path:
        return self.root / "converted"

    @property
    def enhanced(self) -> Path:
        return self.root / "enhanced"

    @property
    def work(self) -> Path:
        return self.root / "work"

    @property
    def clips(self) -> Path:
        return self.root / "clips"

    @property
    def exports(self) -> Path:
        return self.root / "exports"

    @property
    def logs(self) -> Path:
        return self.root / "logs"

    @property
    def tmp(self) -> Path:
        return self.root / "tmp"

    @property
    def db_file(self) -> Path:
        return self.root / "portal.db"

    def ensure(self) -> None:
        for d in (self.raw, self.converted, self.enhanced, self.work, self.clips,
                  self.exports, self.logs, self.tmp):
            d.mkdir(parents=True, exist_ok=True)

    # per-source files ------------------------------------------------------
    def converted_wav(self, source_id: int) -> Path:
        return self.converted / f"{source_tag(source_id)}.wav"

    def enhanced_wav(self, source_id: int) -> Path:
        return self.enhanced / f"{source_tag(source_id)}.wav"

    def work_dir(self, source_id: int) -> Path:
        return self.work / source_tag(source_id)

    def clip_dir(self, source_id: int) -> Path:
        return self.clips / source_tag(source_id)

    def clip_wav(self, source_id: int, idx: int) -> Path:
        return self.clip_dir(source_id) / f"{source_tag(source_id)}_{idx:05d}.wav"

    # relative <-> absolute -------------------------------------------------
    def rel(self, path: Path) -> str:
        return Path(path).resolve().relative_to(self.root).as_posix()

    def abs(self, rel_path: str) -> Path:
        p = (self.root / rel_path).resolve()
        if p != self.root and self.root not in p.parents:
            raise ValueError(f"path escapes DATA_ROOT: {rel_path}")
        return p

    def unique_raw_path(self, speaker: str, filename: str, stamp: str) -> Path:
        """Never overwrite an upload: data/raw/<speaker>/<stamp>_<name>[ _n].<ext>"""
        stem, ext = Path(filename).stem, Path(filename).suffix.lower()
        folder = self.raw / slugify(speaker)
        folder.mkdir(parents=True, exist_ok=True)
        candidate = folder / f"{stamp}_{slugify(stem, 80)}{ext}"
        n = 1
        while candidate.exists():
            candidate = folder / f"{stamp}_{slugify(stem, 80)}_{n}{ext}"
            n += 1
        return candidate

    def remove_derived(self, source_id: int) -> None:
        """Delete everything derived from a source (never the original upload)."""
        for f in (self.converted_wav(source_id), self.enhanced_wav(source_id)):
            f.unlink(missing_ok=True)
        for d in (self.work_dir(source_id), self.clip_dir(source_id)):
            shutil.rmtree(d, ignore_errors=True)
