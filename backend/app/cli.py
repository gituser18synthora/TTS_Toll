"""Command-line helpers.

  python -m app.cli init              create DATA_ROOT folders + database
  python -m app.cli doctor            show config, devices and installed components
  python -m app.cli download-models   pre-fetch every model the config uses into models/ (--all: CPU + GPU)
  python -m app.cli check-asr --lang hi   load the secondary ASR (IndicConformer) and transcribe one clip
  python -m app.cli dump-numbers --lang ta   numbers as the portal spells them -> TSV for native speakers
  python -m app.cli pretrain-check <export>  validate an export before training (exit 1 on errors)
  python -m app.cli worker            run the job worker (same as python -m app.worker)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from .config import config_path, configure_model_cache, get_settings
from .db import create_schema, init_engine
from .devices import capability_report, resolve_device
from .paths import DataPaths


def cmd_init(_args) -> None:
    settings = get_settings()
    paths = DataPaths.from_settings(settings)
    paths.ensure()
    init_engine(paths.db_file)
    create_schema()
    print(f"DATA_ROOT ready: {paths.root}\ndatabase: {paths.db_file}")


def cmd_doctor(_args) -> None:
    settings = get_settings()
    configure_model_cache(settings)
    print(f"config:     {config_path()}")
    print(f"data_root:  {settings.data_root}")
    print(f"models_dir: {settings.models_dir}")
    print(f"device:     {settings.device} -> {resolve_device(settings.device)}")
    for name, ok in capability_report(settings).items():
        print(f"  {'ok ' if ok else '-- '} {name}")
    if not capability_report(settings)["ffmpeg"]:
        sys.exit("ffmpeg not found on PATH")


def cmd_download(args) -> None:
    """Fetch every model the current config can use, so the server can run offline afterwards."""
    settings = get_settings()
    configure_model_cache(settings)
    device = resolve_device(settings.device)
    gpu = device == "cuda" or args.all
    cpu = device == "cpu" or args.all

    p, sec = settings.asr.primary, settings.asr.secondary
    sizes = set()
    if p.engine == "faster-whisper":
        sizes |= ({p.model_gpu} if gpu else set()) | ({p.model_cpu} if cpu else set())
    if sec.engine != "none":
        sizes |= ({sec.whisper_gpu} if gpu else set()) | ({sec.whisper_cpu} if cpu else set())
    if sizes:
        from faster_whisper.utils import download_model

        for size in sorted(sizes):
            print(f"faster-whisper '{size}' -> {settings.models_dir / 'whisper'}")
            download_model(size, cache_dir=str(settings.models_dir / "whisper"))
    try:
        from .pipeline.vad import _load_silero

        _load_silero()
        print("silero-vad: bundled with the pip package (ok)")
    except ImportError as exc:
        print(f"silero-vad unavailable ({exc}); energy VAD will be used")
    if settings.denoise.engine == "deepfilternet":
        from .pipeline.denoise import download_model as download_df

        print(f"DeepFilterNet3 -> {download_df(settings.models_dir)}")
    if settings.quality.engine == "dnsmos":
        from .pipeline.quality import download_dnsmos

        print(f"DNSMOS -> {download_dnsmos(settings.models_dir)}")
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import GatedRepoError

    def gated(label: str, repo: str, fallback: str) -> None:
        if not os.environ.get("HF_TOKEN"):
            print(f"{label}: skipped (no HF_TOKEN in .env); {fallback}")
            return
        try:
            print(f"{label} -> {snapshot_download(repo)}")
        except GatedRepoError:
            print(f"{label}: no access. Accept the terms at https://huggingface.co/{repo} "
                  f"with the account of your HF_TOKEN, then run this again.")

    tagger = settings.quality.tagger
    if tagger.enabled is True or (tagger.enabled == "auto" and gpu):
        print(f"audio tagger {tagger.model} -> {snapshot_download(tagger.model)}")
    if sec.engine == "indicconformer":
        gated("IndicConformer", sec.indicconformer.model, "the whisper fallback will be used")
    # speaker check + clustering diarization fallback; not gated
    print(f"speaker embeddings {settings.speaker_check.model} -> {snapshot_download(settings.speaker_check.model)}")
    if settings.diarization.engine == "pyannote":
        gated("diarization", settings.diarization.model, "multi-speaker files use the clustering fallback")
    print("done")


def cmd_check_asr(args) -> None:
    """Load the secondary ASR for a language and transcribe one clip (or --file)."""
    import time

    import soundfile as sf
    from sqlalchemy import select

    from .db import session_scope
    from .models import Clip
    from .pipeline import asr, audio_ops

    settings = get_settings()
    configure_model_cache(settings)
    paths = DataPaths.from_settings(settings)
    init_engine(paths.db_file)
    if args.file:
        audio_path = args.file
    else:
        with session_scope() as s:
            clip = s.scalars(select(Clip).where(Clip.language == args.lang).order_by(Clip.id)).first()
        if clip is None:
            sys.exit(f"no clip with language {args.lang!r} in the database; pass --file some.wav")
        audio_path = str(paths.abs(clip.clip_path))
        print(f"clip {clip.id}: {clip.text}")
    x, sr = sf.read(audio_path, dtype="float32", always_2d=True)
    x16 = audio_ops.resample_poly_to(x.mean(axis=1), sr, 16000)
    print(f"HF_TOKEN set: {bool(os.environ.get('HF_TOKEN'))}")
    t0 = time.time()
    engine = asr.get_secondary_asr(settings, args.lang)
    if engine is None:
        sys.exit("no secondary ASR available")
    reason = asr.secondary_fallback_reason(settings)
    print(f"engine: {engine.name}" + (f"   (IndicConformer unavailable: {reason})" if reason else "")
          + f"   loaded in {time.time() - t0:.1f}s")
    hint = args.lang if isinstance(engine, asr.IndicConformerASR) else settings.whisper_language(args.lang)
    t0 = time.time()
    result = engine.transcribe(x16, hint)
    print(f"transcript ({time.time() - t0:.1f}s for {len(x16) / 16000:.1f}s audio): {result.text}")


NUMBER_SAMPLES: list[tuple[str, str]] = [
    *[("0-20", str(n)) for n in range(21)],
    *[("tens", str(n)) for n in range(30, 100, 10)],
    ("hundred", "100"), ("thousand", "1000"), ("lakh", "1,00,000"), ("crore", "1,00,00,000"),
    ("decimal", "2.5"), ("decimal", "1.87"), ("year", "2024"), ("currency", "₹500"),
    # compound forms: where joining rules (sandhi) are most likely to be wrong
    ("compound", "21"), ("compound", "45"), ("compound", "99"), ("compound", "101"), ("compound", "150"),
    ("compound", "1500"), ("compound", "2000"), ("compound", "2,50,000"), ("year", "1947"), ("percent", "50%"),
]
NUMBER_NOTES = {
    "1500": "standalone 4-digit numbers 1100-2099 are read like years",
    "2024": "read as a year", "1947": "read as a year",
    "1.87": "digits after the point are read one by one",
}


def cmd_dump_numbers(args) -> None:
    """Write how the portal spells numbers for one language, for native-speaker review."""
    from .pipeline.text_norm import normalize_transcript
    from .pipeline.translit import to_roman

    settings = get_settings()
    lang = args.lang
    roman = lang in ("hi", "hinglish")
    scheme = args.scheme or settings.text.roman_scheme
    out = Path(args.out) if args.out else DataPaths.from_settings(settings).root / "reports" / f"numbers_{lang}.tsv"
    out.parent.mkdir(parents=True, exist_ok=True)
    header = ["#", "category", "written", "spoken (as the portal writes it)"]
    if roman:
        header.append(f"roman export ({scheme})")
    header += ["note", "correct? (y/n)", "correction"]
    rows = []
    for i, (category, written) in enumerate(NUMBER_SAMPLES, 1):
        spoken = normalize_transcript(written, lang, settings).text  # same code path as real transcripts
        row = [str(i), category, written, spoken]
        if roman:
            row.append(to_roman(spoken, scheme))
        row += [NUMBER_NOTES.get(written, ""), "", ""]
        rows.append(row)
    with open(out, "w", encoding="utf-8") as fh:
        for row in [header, *rows]:
            fh.write("\t".join(row) + "\n")
    width = max(len(r[2]) for r in rows)
    for r in rows:
        extra = f"   | {r[4]}" if roman else ""
        print(f"{r[0]:>3}  {r[2]:<{width}}  {r[3]}{extra}")
    print(f"\n{len(rows)} numbers -> {out}\n(tab-separated: opens in Excel / Google Sheets; reviewers fill the last two columns)")


def cmd_pretrain_check(args) -> None:
    """Validate an export (name, folder or .zip) before training. Exit code 1 on any error."""
    import shutil
    import tempfile
    import zipfile

    from .export.pretrain_check import check_export, format_report

    settings = get_settings()
    target = Path(args.export)
    if not target.exists():
        target = DataPaths.from_settings(settings).exports / args.export
    if not target.exists():
        sys.exit(f"export not found: {args.export} (give a folder, a .zip, or an export name under data/exports/)")
    tmp = None
    try:
        root = target
        if target.suffix == ".zip":
            tmp = Path(tempfile.mkdtemp(prefix="pretrain_check_"))
            with zipfile.ZipFile(target) as zf:
                zf.extractall(tmp)
            tops = [p for p in tmp.iterdir() if p.is_dir()]
            root = tops[0] if len(tops) == 1 else tmp
        seg = settings.segmentation
        rep = check_export(root, min_duration=args.min_duration if args.min_duration is not None else seg.min_duration,
                           max_duration=args.max_duration if args.max_duration is not None else seg.hard_max_duration,
                           min_speaker_minutes=args.min_speaker_minutes, verify_checksums=not args.no_checksums)
        print(format_report(rep, args.max_examples))
        if args.json:
            Path(args.json).write_text(json.dumps(rep.to_json(), indent=2, ensure_ascii=False), encoding="utf-8")
            print(f"report -> {args.json}")
    finally:
        if tmp:
            shutil.rmtree(tmp, ignore_errors=True)
    sys.exit(1 if rep.errors else 0)


def cmd_worker(_args) -> None:
    from .worker.runner import main

    main()


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m app.cli")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("init").set_defaults(fn=cmd_init)
    sub.add_parser("doctor").set_defaults(fn=cmd_doctor)
    dl = sub.add_parser("download-models")
    dl.add_argument("--all", action="store_true", help="download models for both CPU and GPU setups")
    dl.set_defaults(fn=cmd_download)
    ck = sub.add_parser("check-asr", help="verify the secondary ASR (IndicConformer) on one clip")
    ck.add_argument("--lang", default="hi", help="language code (default: hi)")
    ck.add_argument("--file", help="audio file to transcribe instead of a clip from the database")
    ck.set_defaults(fn=cmd_check_asr)
    dn = sub.add_parser("dump-numbers", help="write how numbers are spelled for one language (native-speaker check)")
    dn.add_argument("--lang", required=True, choices=["hi", "hinglish", "ta", "te", "ml", "en"])
    dn.add_argument("--out", help="output .tsv (default: DATA_ROOT/reports/numbers_<lang>.tsv)")
    dn.add_argument("--scheme", choices=["hinglish", "iso", "iast", "itrans"], help="roman scheme for the hi column")
    dn.set_defaults(fn=cmd_dump_numbers)
    pc = sub.add_parser("pretrain-check", help="validate an export before training (exit 1 on errors)")
    pc.add_argument("export", help="export name (data/exports/<name>), export folder, or the .zip")
    pc.add_argument("--min-speaker-minutes", type=float, default=30.0, help="warn below this per speaker (default 30)")
    pc.add_argument("--min-duration", type=float, help="default: segmentation.min_duration")
    pc.add_argument("--max-duration", type=float, help="default: segmentation.hard_max_duration")
    pc.add_argument("--no-checksums", action="store_true", help="skip SHA256SUMS verification")
    pc.add_argument("--json", help="also write the full report as JSON")
    pc.add_argument("--max-examples", type=int, default=5)
    pc.set_defaults(fn=cmd_pretrain_check)
    sub.add_parser("worker").set_defaults(fn=cmd_worker)
    sub.add_parser("config").set_defaults(fn=lambda _a: print(json.dumps(get_settings().model_dump(), indent=2, default=str)))
    args = parser.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
