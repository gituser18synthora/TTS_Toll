"""Export approved clips as a self-contained training folder (+ zip).

Layout of data/exports/<name>/ (re-running an export with the same name rebuilds it):
  wavs/<speaker>_src_XXXXX_YYYYY.wav   24 kHz mono PCM16
  train.jsonl        {"audio": "wavs/...wav", "text": "<speaker>: <transcript>"}  (Orpheus / Unsloth)
  manifest.jsonl     same rows with full metadata (speaker, gender, language, script, duration, ids, ...)
  metadata.csv       LJSpeech-style: id|text|normalized_text|speaker|language|script   (optional)
  hf_dataset/        HuggingFace `datasets.save_to_disk` (optional)
  stats.json         totals per speaker / language / script
  export.log         what was exported, skipped or fixed
  README.txt         format description
  SHA256SUMS         `sha256sum -c SHA256SUMS` on the server verifies the copy

Script option (Hindi / Hinglish clips only; other languages always export as stored):
  devanagari  the stored canonical text (English words stay in Latin script)
  roman       romanized with translit.to_roman (text.roman_scheme)
  both        one row per script for the same wav file
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf
from sqlalchemy import Select, case, func, select
from sqlalchemy.orm import Session

from ..config import Settings
from ..db import session_scope
from ..models import Clip, Export, Speaker, utcnow
from ..paths import DataPaths, slugify, source_tag
from ..pipeline import audio_ops
from ..pipeline.text_norm import foreign_letter_ratio
from ..pipeline.translit import to_roman

ROMANIZABLE = {"hi", "hinglish"}


def approved_query(speaker_ids: list[int] | None, languages: list[str] | None) -> Select:
    q = select(Clip).where(Clip.status == "approved")
    if speaker_ids:
        q = q.where(Clip.speaker_id.in_(speaker_ids))
    if languages:
        q = q.where(Clip.language.in_(languages))
    return q.order_by(Clip.speaker_id, Clip.source_file_id, Clip.idx)


def preview(session: Session, speaker_ids: list[int] | None, languages: list[str] | None) -> dict:
    """Counts/hours of approved clips matching the filters (no files touched)."""
    q = (select(Speaker.name, Clip.language, func.count(Clip.id), func.sum(Clip.duration),
                func.sum(case((func.trim(func.coalesce(Clip.text, "")) == "", 1), else_=0)))
         .join(Speaker, Speaker.id == Clip.speaker_id)
         .where(Clip.status == "approved"))
    if speaker_ids:
        q = q.where(Clip.speaker_id.in_(speaker_ids))
    if languages:
        q = q.where(Clip.language.in_(languages))
    rows = session.execute(q.group_by(Speaker.name, Clip.language).order_by(Speaker.name, Clip.language)).all()
    groups = [{"speaker": s, "language": lang, "clips": n, "hours": round((d or 0) / 3600, 3),
               "empty_text": int(e or 0)} for s, lang, n, d, e in rows]
    return {
        "clips": sum(g["clips"] for g in groups),
        "hours": round(sum(g["hours"] for g in groups), 3),
        "empty_text": sum(g["empty_text"] for g in groups),
        "romanizable_clips": sum(g["clips"] for g in groups if g["language"] in ROMANIZABLE),
        "groups": groups,
    }


def text_variants(text: str, language: str, script: str, scheme: str) -> list[tuple[str, str]]:
    """[(text, script_label)] rows for one clip."""
    if language not in ROMANIZABLE:
        return [(text, "native")]
    if script == "devanagari":
        return [(text, "devanagari")]
    roman = to_roman(text, scheme)
    if script == "roman" or roman == text:
        return [(roman, "roman")]
    return [(text, "devanagari"), (roman, "roman")]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _place_audio(src: Path, dst: Path, sample_rate: int, verify: bool) -> str | None:
    """Copy (hard-link when possible) a clip into the export. Returns a note if it had to be fixed."""
    if verify:
        info = sf.info(str(src))
        if info.samplerate != sample_rate or info.channels != 1 or info.subtype != "PCM_16":
            x, sr = audio_ops.read_wav(src)
            x = audio_ops.resample_poly_to(x, sr, sample_rate)
            peak = float(np.max(np.abs(x))) if len(x) else 0.0
            if peak > 1.0:
                x = x / peak
            audio_ops.write_wav(dst, x, sample_rate)
            return f"fixed format ({info.samplerate} Hz, {info.channels} ch, {info.subtype}) -> {sample_rate} Hz mono PCM_16"
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)
    return None


def write_hf_dataset(rows: list[dict], root: Path, out_dir: Path, tmp_dir: Path, sample_rate: int) -> int:
    """HuggingFace `save_to_disk` dataset with the audio bytes embedded (portable, no paths).

    The Arrow data is streamed to disk in the `datasets.Audio` storage layout, so building it
    needs neither torchcodec nor all audio in memory. On the training machine:
        ds = datasets.load_from_disk("hf_dataset")   # audio decodes at 24 kHz (needs torchcodec)
    """
    import pyarrow as pa
    from datasets import Audio, Dataset, DatasetInfo, Features, Value

    features = Features({
        "audio": Audio(sampling_rate=sample_rate), "text": Value("string"), "transcript": Value("string"),
        "speaker": Value("string"), "gender": Value("string"), "language": Value("string"),
        "script": Value("string"), "duration": Value("float32"), "id": Value("string"),
    })
    schema = features.arrow_schema
    tmp_dir.mkdir(parents=True, exist_ok=True)
    stream = tmp_dir / "data.arrow"
    try:
        with pa.OSFile(str(stream), "wb") as sink, pa.ipc.new_stream(sink, schema) as writer:
            for r in rows:
                audio = {"bytes": (root / r["audio"]).read_bytes(), "path": Path(r["audio"]).name}
                writer.write_batch(pa.record_batch([
                    pa.array([audio], type=schema.field("audio").type),
                    pa.array([f"{r['speaker']}: {r['text']}"]), pa.array([r["text"]]), pa.array([r["speaker"]]),
                    pa.array([r["gender"]]), pa.array([r["language"]]), pa.array([r["script"]]),
                    pa.array([r["duration"]], type=pa.float32()), pa.array([r["row_id"]]),
                ], schema=schema))
        ds = Dataset.from_file(str(stream), info=DatasetInfo(features=features))
        ds.save_to_disk(str(out_dir))
        return len(ds)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


README = """Orpheus TTS dataset export: {name}
Created: {created} UTC
Clips: {clips} ({rows} training rows)   Total audio: {hours:.2f} h   Audio: 24 kHz mono 16-bit PCM WAV
Hindi / Hinglish script: {script}

train.jsonl     one JSON object per line, ready for Orpheus / Unsloth fine-tuning:
                  {{"audio": "wavs/<file>.wav", "text": "<speaker_name>: <transcript>"}}
                audio paths are relative to this folder. With script "both", Hindi/Hinglish
                clips appear twice (Devanagari row + romanized row, same wav).
manifest.jsonl  the same rows plus speaker, gender, language, script, duration and source ids.
metadata.csv    LJSpeech-style fallback: id|text|normalized_text|speaker|language|script
                (wav file = wavs/<id>.wav)
hf_dataset/     (if selected) HuggingFace datasets.save_to_disk folder, audio embedded:
                  ds = datasets.load_from_disk("hf_dataset")
                columns: audio, text ("<speaker>: <transcript>"), transcript, speaker, gender,
                language, script, duration, id
stats.json      totals per speaker / language / script.
SHA256SUMS      verify after copying:  sha256sum -c SHA256SUMS

Load train.jsonl as a HuggingFace dataset on the training server:
  from datasets import load_dataset, Audio
  ds = load_dataset("json", data_files="train.jsonl", split="train")
  ds = ds.map(lambda r: {{"audio": "<this folder>/" + r["audio"]}})
  ds = ds.cast_column("audio", Audio(sampling_rate=24000))
"""


def run_export(ctx, settings: Settings, export_id: int) -> None:
    paths = DataPaths.from_settings(settings)
    with session_scope(write=True) as s:
        exp = s.get(Export, export_id)
        exp.status, exp.error = "running", None
        params = json.loads(exp.params or "{}")
        name = slugify(exp.name, 64)
    formats = set(params.get("formats") or ["jsonl"])
    verify = bool(params.get("verify_audio", True))
    make_zip = bool(params.get("make_zip", True))
    script = params.get("script") or "devanagari"
    scheme = settings.text.roman_scheme
    sr = settings.audio.sample_rate

    out_root = paths.exports / name
    build = paths.exports / f".{name}.building"
    shutil.rmtree(build, ignore_errors=True)
    (build / "wavs").mkdir(parents=True)
    log_lines: list[str] = []

    def note(msg: str, warn: bool = False) -> None:
        log_lines.append(f"[{utcnow():%Y-%m-%d %H:%M:%S} UTC] {'WARNING: ' if warn else ''}{msg}")
        (ctx.warn if warn else ctx.log)(msg)

    note(f"export '{name}': speakers={params.get('speaker_ids') or 'all'}, "
         f"languages={params.get('languages') or 'all'}, formats={sorted(formats)}, script={script}"
         f"{f' ({scheme})' if script != 'devanagari' else ''}, verify_audio={verify}, zip={make_zip}")

    with session_scope() as s:
        clips = s.scalars(approved_query(params.get("speaker_ids"), params.get("languages"))).unique().all()
        speakers = {sp.id: sp for sp in s.scalars(select(Speaker))}
    note(f"{len(clips)} approved clips match")

    copy_report = ctx.sub(0.0, 0.5, "copying audio")
    records: list[dict] = []  # one per clip (unique audio)
    rows: list[dict] = []     # one per training example
    seen_audio: dict[str, int] = {}
    for n, clip in enumerate(clips, 1):
        spk = speakers[clip.speaker_id]
        text = (clip.text or "").strip()
        if not text:
            note(f"skip clip {clip.id}: empty transcript", warn=True)
            continue
        if clip.audio_hash and clip.audio_hash in seen_audio:
            note(f"skip clip {clip.id}: identical audio to clip {seen_audio[clip.audio_hash]}", warn=True)
            continue
        if foreign_letter_ratio(text, clip.language) > 0.5:
            note(f"skip clip {clip.id}: transcript is mostly in the wrong script for {clip.language}", warn=True)
            continue
        if clip.audio_hash:
            seen_audio[clip.audio_hash] = clip.id
        src = paths.abs(clip.clip_path)
        if not src.exists():
            note(f"skip clip {clip.id}: audio file missing ({clip.clip_path})", warn=True)
            continue
        uid = f"{slugify(spk.name, 32)}_{source_tag(clip.source_file_id)}_{clip.idx:05d}"
        fixed = _place_audio(src, build / "wavs" / f"{uid}.wav", sr, verify)
        if fixed:
            note(f"clip {clip.id}: {fixed}", warn=True)
        record = {
            "id": uid, "audio": f"wavs/{uid}.wav", "speaker": spk.name, "gender": spk.gender,
            "language": clip.language, "duration": round(clip.duration, 3), "clip_id": clip.id,
            "source_file_id": clip.source_file_id, "start_time": clip.start_time, "end_time": clip.end_time,
            "quality_score": clip.quality_score, "asr_agreement": clip.asr_agreement,
            "speaker_similarity": clip.speaker_similarity, "needs_review": clip.needs_review, "flags": clip.flags(),
        }
        records.append(record)
        for variant, label in text_variants(text, clip.language, script, scheme):
            rows.append({**record, "text": variant, "script": label,
                         "row_id": uid if label != "roman" or script == "roman" else f"{uid}_roman"})
        copy_report(n / max(len(clips), 1), f"{n}/{len(clips)}")
        ctx.check_cancel()

    # manifests --------------------------------------------------------------
    with open(build / "train.jsonl", "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps({"audio": r["audio"], "text": f"{r['speaker']}: {r['text']}"}, ensure_ascii=False) + "\n")
    with open(build / "manifest.jsonl", "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps({k: v for k, v in r.items() if k != "row_id"}, ensure_ascii=False) + "\n")
    if "csv" in formats:
        with open(build / "metadata.csv", "w", encoding="utf-8") as fh:
            for r in rows:
                t = r["text"].replace("|", " ").replace("\n", " ")
                fh.write(f"{r['id']}|{t}|{t}|{r['speaker']}|{r['language']}|{r['script']}\n")
    if "hf" in formats:
        hf_report = ctx.sub(0.5, 0.1, "hf dataset")
        n_hf = write_hf_dataset(rows, build, build / "hf_dataset", paths.tmp / f"hf_{export_id}", sr)
        hf_report(1.0)
        note(f"hf_dataset: {n_hf} rows (load with datasets.load_from_disk)")

    # stats ------------------------------------------------------------------------
    def bucketize(items: list[dict], key) -> dict:
        out: dict = defaultdict(lambda: {"clips": 0, "seconds": 0.0})
        for r in items:
            out[key(r)]["clips"] += 1
            out[key(r)]["seconds"] += r["duration"]
        return {k: {"clips": v["clips"], "hours": round(v["seconds"] / 3600, 3)} for k, v in sorted(out.items())}

    total_sec = sum(r["duration"] for r in records)
    stats = {
        "name": name, "created_at": utcnow().isoformat() + "Z", "clips": len(records), "rows": len(rows),
        "hours": round(total_sec / 3600, 3), "skipped": len(clips) - len(records), "sample_rate": sr,
        "script": script, "roman_scheme": scheme if script != "devanagari" else None,
        "per_speaker": bucketize(records, lambda r: r["speaker"]),
        "per_language": bucketize(records, lambda r: r["language"]),
        "per_speaker_language": bucketize(records, lambda r: f"{r['speaker']}/{r['language']}"),
        "rows_per_script": {k: v["clips"] for k, v in bucketize(rows, lambda r: r["script"]).items()},
        "params": params,
    }
    (build / "stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    for key, v in stats["per_speaker_language"].items():
        note(f"  {key}: {v['clips']} clips, {v['hours']:.2f} h")
    note(f"total: {len(records)} clips / {len(rows)} rows, {total_sec / 3600:.2f} h, skipped {stats['skipped']}")
    (build / "README.txt").write_text(README.format(
        name=name, created=f"{utcnow():%Y-%m-%d %H:%M}", clips=len(records), rows=len(rows),
        hours=total_sec / 3600, script=script if script == "devanagari" else f"{script} ({scheme})"),
        encoding="utf-8")
    (build / "export.log").write_text("\n".join(log_lines) + "\n", encoding="utf-8")

    # checksums ----------------------------------------------------------------------
    hash_report = ctx.sub(0.6, 0.15, "checksums")
    files = sorted(p for p in build.rglob("*") if p.is_file())
    with open(build / "SHA256SUMS", "w", encoding="utf-8") as fh:
        for n, p in enumerate(files, 1):
            fh.write(f"{_sha256(p)}  {p.relative_to(build).as_posix()}\n")
            if n % 50 == 0:
                hash_report(n / len(files))
                ctx.check_cancel()

    # swap into place (the previous export with this name is replaced) --------------------
    old = paths.exports / f".{name}.old"
    shutil.rmtree(old, ignore_errors=True)
    if out_root.exists():
        os.replace(out_root, old)
    os.replace(build, out_root)
    shutil.rmtree(old, ignore_errors=True)

    zip_rel = None
    if make_zip:
        zip_report = ctx.sub(0.75, 0.25, "zipping")
        zip_path = paths.exports / f"{name}.zip"
        part = zip_path.with_name(zip_path.name + ".part")
        all_files = sorted(p for p in out_root.rglob("*") if p.is_file())
        with zipfile.ZipFile(part, "w", allowZip64=True) as zf:
            for n, p in enumerate(all_files, 1):
                stored = p.suffix in (".wav", ".arrow")
                zf.write(p, f"{name}/{p.relative_to(out_root).as_posix()}",
                         compress_type=zipfile.ZIP_STORED if stored else zipfile.ZIP_DEFLATED)
                if n % 50 == 0:
                    zip_report(n / len(all_files))
                    ctx.check_cancel()
        os.replace(part, zip_path)
        zip_rel = paths.rel(zip_path)
        ctx.log(f"zip: {zip_path.name} ({zip_path.stat().st_size / 1e6:.1f} MB)")

    with session_scope(write=True) as s:
        exp = s.get(Export, export_id)
        exp.status = "done"
        exp.out_dir = paths.rel(out_root)
        exp.zip_path = zip_rel
        exp.clip_count = len(records)
        exp.total_duration = total_sec
        exp.stats = json.dumps(stats, ensure_ascii=False)
        exp.finished_at = utcnow()
