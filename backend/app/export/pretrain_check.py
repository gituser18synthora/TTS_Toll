"""Validate an export folder before training. Needs only the folder (no portal database).

ERRORs (exit code 1): missing/invalid train.jsonl rows, audio not 24 kHz mono 16-bit,
duration outside [min_duration, hard_max_duration], empty text or missing "<speaker>: "
prefix, text mostly in the wrong script, clips exported while flagged wrong_script or
needs_review, duplicate (audio, text) rows, the same audio under two file names, and
SHA256SUMS mismatches.
WARNINGs: a few foreign letters in a transcript, speakers with less than
--min-speaker-minutes of audio, missing manifest / checksums.
"""
from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import soundfile as sf

from ..pipeline.text_norm import foreign_letter_ratio

SAMPLE_RATE = 24000


@dataclass
class Issue:
    level: str   # ERROR | WARN
    code: str
    message: str
    row: int | None = None


@dataclass
class Report:
    root: str
    rows: int = 0
    issues: list[Issue] = field(default_factory=list)
    speakers: dict = field(default_factory=dict)  # name -> {minutes, clips, rows, languages: {lang: minutes}}

    @property
    def errors(self) -> list[Issue]:
        return [i for i in self.issues if i.level == "ERROR"]

    @property
    def warnings(self) -> list[Issue]:
        return [i for i in self.issues if i.level == "WARN"]

    def add(self, level: str, code: str, message: str, row: int | None = None) -> None:
        self.issues.append(Issue(level, code, message, row))

    def to_json(self) -> dict:
        return {"root": self.root, "rows": self.rows, "ok": not self.errors,
                "errors": len(self.errors), "warnings": len(self.warnings),
                "speakers": self.speakers, "issues": [i.__dict__ for i in self.issues]}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check_export(root: Path, *, min_duration: float, max_duration: float, min_speaker_minutes: float = 30.0,
                 verify_checksums: bool = True) -> Report:
    root = root.resolve()
    rep = Report(str(root))
    train = root / "train.jsonl"
    if not train.exists():
        rep.add("ERROR", "no_train_jsonl", "train.jsonl not found")
        return rep
    rows: list[dict | None] = []
    for n, line in enumerate(train.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except ValueError as exc:
            rep.add("ERROR", "bad_json", f"line {n}: {exc}", n)
            rows.append(None)
    rep.rows = len(rows)
    manifest: list[dict] | None = None
    if (root / "manifest.jsonl").exists():
        manifest = [json.loads(x) for x in (root / "manifest.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
        if len(manifest) != len(rows):
            rep.add("ERROR", "manifest_mismatch", f"manifest.jsonl has {len(manifest)} rows, train.jsonl {len(rows)}")
            manifest = None
    else:
        rep.add("WARN", "no_manifest", "manifest.jsonl missing: language / flags cannot be checked")

    sha_by_path: dict[str, str] = {}
    paths_by_sha: dict[str, set[str]] = defaultdict(set)
    seen_pairs: dict[tuple[str, str], int] = {}
    seconds: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    audio_seen_by_speaker: dict[str, set[str]] = defaultdict(set)
    row_count: dict[str, int] = defaultdict(int)

    for i, row in enumerate(rows, 1):
        if row is None:
            continue
        if set(row) != {"audio", "text"}:
            rep.add("ERROR", "bad_keys", f"keys {sorted(row)} (expected audio, text)", i)
        meta = manifest[i - 1] if manifest else {}
        text = str(row.get("text") or "")
        speaker = meta.get("speaker") or (text.split(":", 1)[0] if ":" in text else "")
        prefix = f"{speaker}: "
        if not speaker or not text.startswith(prefix):
            rep.add("ERROR", "missing_prefix", f"text does not start with '{prefix}': {text[:60]!r}", i)
            body = text.split(":", 1)[1].strip() if ":" in text else text
        else:
            body = text[len(prefix):].strip()
        if not body:
            rep.add("ERROR", "empty_text", "transcript is empty", i)
        if "\n" in text or "\r" in text:
            rep.add("ERROR", "newline", "transcript contains a line break", i)
        language = meta.get("language")
        if language and body:
            ratio = foreign_letter_ratio(body, language, roman=meta.get("script") == "roman")
            if ratio > 0.5:
                rep.add("ERROR", "wrong_script", f"{ratio:.0%} of letters are outside the {language} script: {body[:50]!r}", i)
            elif ratio > 0:
                rep.add("WARN", "foreign_letters", f"{ratio:.0%} foreign letters ({language}): {body[:50]!r}", i)
        if meta.get("needs_review"):
            rep.add("ERROR", "needs_review", f"clip {meta.get('clip_id')} was still flagged needs_review", i)
        if "wrong_script" in (meta.get("flags") or []):
            rep.add("ERROR", "wrong_script_flag", f"clip {meta.get('clip_id')} carries the wrong_script flag", i)

        rel = str(row.get("audio") or "")
        wav = (root / rel).resolve()
        if not rel or root not in wav.parents or not wav.exists():
            rep.add("ERROR", "missing_audio", f"audio file not found: {rel!r}", i)
            continue
        if rel not in sha_by_path:
            try:
                info = sf.info(str(wav))
            except Exception as exc:  # noqa: BLE001 - unreadable file is an error, not a crash
                rep.add("ERROR", "bad_audio", f"{rel}: {exc}", i)
                continue
            if (info.samplerate, info.channels, info.subtype) != (SAMPLE_RATE, 1, "PCM_16"):
                rep.add("ERROR", "bad_format", f"{rel}: {info.samplerate} Hz, {info.channels} ch, {info.subtype} "
                                               f"(need {SAMPLE_RATE} Hz mono PCM_16)", i)
            dur = info.frames / info.samplerate
            if not min_duration <= dur <= max_duration:
                rep.add("ERROR", "duration", f"{rel}: {dur:.2f} s outside [{min_duration}, {max_duration}]", i)
            sha_by_path[rel] = _sha256(wav)
            paths_by_sha[sha_by_path[rel]].add(rel)
            if rel not in audio_seen_by_speaker[speaker]:
                audio_seen_by_speaker[speaker].add(rel)
                seconds[speaker][language or "?"] += dur
        sha = sha_by_path[rel]
        if (sha, body) in seen_pairs:
            rep.add("ERROR", "duplicate_row", f"same audio and text as row {seen_pairs[(sha, body)]}", i)
        else:
            seen_pairs[(sha, body)] = i
        row_count[speaker] += 1

    for sha, rels in paths_by_sha.items():
        if len(rels) > 1:
            rep.add("ERROR", "duplicate_audio", f"identical audio in {len(rels)} files: {sorted(rels)[:3]}")

    sums = root / "SHA256SUMS"
    if verify_checksums:
        if not sums.exists():
            rep.add("WARN", "no_checksums", "SHA256SUMS missing: copy integrity not verified")
        else:
            for line in sums.read_text(encoding="utf-8").splitlines():
                if "  " not in line:
                    continue
                digest, rel = line.split("  ", 1)
                path = root / rel
                actual = sha_by_path.get(rel) or (_sha256(path) if path.exists() else None)
                if actual is None:
                    rep.add("ERROR", "checksum_missing_file", f"{rel} listed in SHA256SUMS but missing")
                elif actual != digest:
                    rep.add("ERROR", "checksum_mismatch", f"{rel}: content differs from SHA256SUMS (corrupted copy?)")

    for speaker, langs in sorted(seconds.items()):
        minutes = sum(langs.values()) / 60
        rep.speakers[speaker] = {
            "minutes": round(minutes, 1), "clips": len(audio_seen_by_speaker[speaker]), "rows": row_count[speaker],
            "languages": {k: round(v / 60, 1) for k, v in sorted(langs.items())},
        }
        if minutes < min_speaker_minutes:
            rep.add("WARN", "low_speaker_data",
                    f"speaker {speaker!r} has only {minutes:.1f} min (< {min_speaker_minutes:g} min) of approved audio")
    return rep


def format_report(rep: Report, max_examples: int = 5) -> str:
    out = [f"pretrain-check: {rep.root}", f"rows: {rep.rows}", ""]
    if rep.speakers:
        out.append(f"{'speaker':<24}{'minutes':>9}{'clips':>8}{'rows':>7}   per language (min)")
        for name, s in rep.speakers.items():
            langs = ", ".join(f"{k} {v}" for k, v in s["languages"].items())
            out.append(f"{name[:23]:<24}{s['minutes']:>9}{s['clips']:>8}{s['rows']:>7}   {langs}")
        total = sum(s["minutes"] for s in rep.speakers.values())
        out.append(f"{'TOTAL':<24}{total:>9.1f}{sum(s['clips'] for s in rep.speakers.values()):>8}{rep.rows:>7}")
        out.append("")
    by_code: dict[tuple[str, str], list[Issue]] = defaultdict(list)
    for issue in rep.issues:
        by_code[(issue.level, issue.code)].append(issue)
    for (level, code), items in sorted(by_code.items(), key=lambda kv: (kv[0][0] != "ERROR", kv[0][1])):
        out.append(f"{level:<5} {code} ({len(items)})")
        for it in items[:max_examples]:
            out.append(f"        {'row ' + str(it.row) + ': ' if it.row else ''}{it.message}")
        if len(items) > max_examples:
            out.append(f"        … {len(items) - max_examples} more")
    out.append("")
    out.append(f"RESULT: {'FAIL' if rep.errors else 'OK'}  ({len(rep.errors)} errors, {len(rep.warnings)} warnings)")
    return "\n".join(out)
