"""Per-source processing pipeline: resumable, idempotent, step by step.

Steps (each checkpointed in source_files.steps_done):
  convert  -> data/converted/src_XXXXX.wav (24 kHz mono 16-bit)
  denoise  -> data/enhanced/src_XXXXX.wav (DeepFilterNet; skipped with a warning if unavailable)
  vad      -> data/work/src_XXXXX/vad.npz (per-frame speech probabilities, on the enhanced audio)
  diarize  -> multi-speaker files only: speaker turns + overlap + sample clips (diarization.json)
  assign   -> multi-speaker files only: pauses (status needs_speakers) until every label is named
  clips    -> data/clips/src_XXXXX/*.wav + clip rows (cut at pauses, one speaker per clip, -23 LUFS)
  quality  -> DNSMOS (or SNR) score per clip, auto-reject low_quality, music/babble flags
  asr      -> primary transcript per clip (resumes per clip)
  asr2     -> secondary transcript + WER agreement per clip (resumes per clip)
  speakers -> speaker embeddings + consistency check (speaker_mismatch) across the speaker's clips
  dedup    -> exact / near-identical duplicates against the whole database
  finalize -> summary (always re-run)

`reset=True` (Reprocess) deletes every derived file and clip row first; the original
upload is never touched. A failed / interrupted run resumes at the first unfinished step.
A step that is still missing on resume is skipped when a later step that consumed its
output already ran (e.g. denoise for a file segmented before denoising existed).

`retranscribe()` re-runs asr + asr2 on the existing clips without re-segmenting, keeping
review status and edited transcripts (see transcripts.py).
"""
from __future__ import annotations

import json
import shutil
import time
from collections import Counter

import numpy as np
import soundfile as sf
from sqlalchemy import delete, func, select, update

from .. import engines
from ..config import Settings
from ..db import session_scope
from ..models import Clip, Job, SourceFile, Speaker, utcnow
from ..paths import DataPaths
from ..worker.jobs import JobContext
from . import asr as asr_mod
from . import audio_ops, convert, dedup, denoise, diarize, quality, segment, transcripts, vad
from . import speakers as speakers_mod

STEPS: list[tuple[str, float]] = [
    ("convert", 0.03),
    ("denoise", 0.06),
    ("vad", 0.02),
    ("diarize", 0.06),
    ("assign", 0.0),
    ("clips", 0.05),
    ("quality", 0.07),
    ("asr", 0.40),
    ("asr2", 0.22),
    ("speakers", 0.05),
    ("dedup", 0.02),
    ("finalize", 0.02),
]
MULTI_SPEAKER_ONLY = {"diarize", "assign"}
INVALIDATED_BY = {"denoise": {"vad", "clips"}, "vad": {"clips"}, "diarize": {"clips"}, "assign": {"clips"}}
ALWAYS_RUN = {"finalize"}
RETRANSCRIBE_STEPS: list[tuple[str, float]] = [("asr", 0.64), ("asr2", 0.32), ("dedup", 0.02), ("finalize", 0.02)]


class AwaitingSpeakers(Exception):
    """Diarization found speakers that have not been named yet; processing pauses here."""


class SourcePipeline:
    def __init__(self, ctx: JobContext, settings: Settings, source_id: int):
        self.ctx = ctx
        self.settings = settings
        self.paths = DataPaths.from_settings(settings)
        self.source_id = source_id

    # -- helpers ---------------------------------------------------------------
    def _source(self) -> SourceFile:
        with session_scope() as s:
            src = s.get(SourceFile, self.source_id)
            if src is None:
                raise RuntimeError(f"source file {self.source_id} not found")
            return src

    def _update_source(self, **values) -> None:
        with session_scope(write=True) as s:
            src = s.get(SourceFile, self.source_id)
            for k, v in values.items():
                setattr(src, k, v)

    def _mark_done(self, step: str) -> None:
        with session_scope(write=True) as s:
            src = s.get(SourceFile, self.source_id)
            done = src.get_steps_done()
            if step not in done:
                done.append(step)
            src.steps_done = json.dumps(done)

    def _delete_clips(self) -> int:
        with session_scope(write=True) as s:
            n = s.execute(delete(Clip).where(Clip.source_file_id == self.source_id)).rowcount
        shutil.rmtree(self.paths.clip_dir(self.source_id), ignore_errors=True)
        return n or 0

    def _audio_for_clips(self, src: SourceFile):
        rel = src.enhanced_path or src.converted_path
        if not rel:
            raise RuntimeError("no converted audio; the convert step did not run")
        return self.paths.abs(rel)

    def _clip_rows(self, *conditions) -> list[tuple]:
        with session_scope() as s:
            return s.execute(select(Clip.id, Clip.clip_path, Clip.duration)
                             .where(Clip.source_file_id == self.source_id, *conditions).order_by(Clip.idx)).all()

    def _clip_audio16k(self, clip_path: str) -> np.ndarray:
        x, sr = audio_ops.read_wav(self.paths.abs(clip_path))
        return audio_ops.resample_poly_to(x, sr, 16000)

    def _diarization(self) -> dict:
        path = self.paths.work_dir(self.source_id) / "diarization.json"
        if not path.exists():
            raise RuntimeError("diarization results missing; Reprocess the file")
        return json.loads(path.read_text(encoding="utf-8"))

    def _run_steps(self, steps: list[tuple[str, float]]) -> None:
        src = self._source()
        done = src.get_steps_done()
        start = 0.0
        for step, weight in steps:
            if step in MULTI_SPEAKER_ONLY and not src.multi_speaker:
                if step not in done:
                    self._mark_done(step)  # nothing to do for single-speaker files
                start += weight
                continue
            if step in done and step not in ALWAYS_RUN:
                self.ctx.log(f"step {step}: already done, skipping")
                start += weight
                continue
            if step not in done and INVALIDATED_BY.get(step, set()) & set(done):
                self.ctx.warn(f"step {step}: skipped, this file was already segmented without it "
                              f"(use Reprocess to apply it)")
                start += weight
                continue
            self._update_source(stage=step)
            report = self.ctx.sub(start, weight, step)
            t0 = time.monotonic()
            getattr(self, f"step_{step}")(report)
            self._mark_done(step)
            self.ctx.log(f"step {step}: done in {time.monotonic() - t0:.1f}s")
            start += weight

    def _finish(self) -> None:
        self.ctx.progress(1.0, "done", force=True)
        self._update_source(status="done", stage="done", progress=100.0, error=None)

    # -- entry points --------------------------------------------------------------
    def run(self, reset: bool = False) -> None:
        if reset:
            n = self._delete_clips()
            self.paths.remove_derived(self.source_id)
            self._update_source(steps_done="[]", converted_path=None, enhanced_path=None, progress=0.0)
            self.ctx.log(f"reprocess: removed {n} old clips and all derived audio (original kept)")
        self._update_source(status="processing", error=None)
        try:
            self._run_steps(STEPS)
        except AwaitingSpeakers as wait:
            self._update_source(status="needs_speakers", stage="assign")
            self.ctx.progress(self.ctx._progress, "waiting for speaker names", force=True)
            self.ctx.log(f"{wait} -> name them on the Files & Jobs page (Assign speakers); processing resumes after that")
            return
        self._finish()

    def retranscribe(self, params: dict) -> None:
        if "clips" not in self._source().get_steps_done():
            raise RuntimeError("this file has no clips yet; process it first")
        if not params.get("cleared"):
            with session_scope(write=True) as s:
                n = s.execute(update(Clip).where(Clip.source_file_id == self.source_id).values(
                    asr_primary_text=None, asr_secondary_text=None, asr_agreement=None)).rowcount
                src = s.get(SourceFile, self.source_id)  # any later resume (retry / process) re-runs ASR
                src.steps_done = json.dumps([x for x in src.get_steps_done() if x not in ("asr", "asr2", "dedup")])
                job = s.get(Job, self.ctx.job_id)
                job.params = json.dumps({**job.get_params(), "cleared": True})  # a resumed job continues
            self.ctx.log(f"re-transcribe: {n} clips (review status and edited transcripts are kept; "
                         f"pending, never-edited transcripts take the new ASR text)")
        self._update_source(status="processing", error=None)
        self._run_steps(RETRANSCRIBE_STEPS)
        self._finish()

    # -- steps ---------------------------------------------------------------------
    def step_convert(self, report) -> None:
        src = self._source()
        original = self.paths.abs(src.original_path)
        if not original.exists():
            raise FileNotFoundError(f"original upload missing: {src.original_path}")
        info = convert.probe(original, self.settings.audio.ffprobe)
        self.ctx.log(f"input: {info['codec']} {info['sample_rate']} Hz, {info['channels']} ch, "
                     f"{(info['duration'] or 0) / 60:.1f} min")
        dst = self.paths.converted_wav(self.source_id)
        convert.to_wav(original, dst, self.settings.audio.sample_rate, ffmpeg=self.settings.audio.ffmpeg,
                       duration=info["duration"], on_progress=report, check_cancel=self.ctx.check_cancel)
        wav = sf.info(str(dst))
        self._update_source(converted_path=self.paths.rel(dst), duration=wav.frames / wav.samplerate)
        self.ctx.log(f"converted to {wav.samplerate} Hz mono PCM16: {wav.frames / wav.samplerate / 60:.2f} min")

    def step_denoise(self, report) -> None:
        denoiser = denoise.get_denoiser(self.settings)
        if denoiser is None:
            self.ctx.warn(f"denoise: {engines.snapshot().get('denoise', 'unavailable')}; "
                          f"clips use the converted audio")
            return
        src = self._source()
        cfg = self.settings.denoise
        work = self.paths.work_dir(self.source_id)
        work.mkdir(parents=True, exist_ok=True)
        wav48, out48 = work / "denoise_in48k.wav", work / "denoise_out48k.wav"
        dst = self.paths.enhanced_wav(self.source_id)
        try:
            convert.to_wav(self.paths.abs(src.original_path), wav48, denoise.DF_SR, ffmpeg=self.settings.audio.ffmpeg,
                           duration=src.duration, on_progress=lambda f: report(0.1 * f),
                           check_cancel=self.ctx.check_cancel)
            denoise.enhance_file(denoiser, wav48, out48, chunk_sec=cfg.chunk_sec, atten_lim_db=cfg.atten_lim_db,
                                 on_progress=lambda f: report(0.1 + 0.8 * f, f"{f * 100:.0f}%"),
                                 check_cancel=self.ctx.check_cancel)
            convert.to_wav(out48, dst, self.settings.audio.sample_rate, ffmpeg=self.settings.audio.ffmpeg,
                           duration=src.duration, on_progress=lambda f: report(0.9 + 0.1 * f),
                           check_cancel=self.ctx.check_cancel)
        finally:
            wav48.unlink(missing_ok=True)
            out48.unlink(missing_ok=True)
        self._update_source(enhanced_path=self.paths.rel(dst))
        lim = f", attenuation limit {cfg.atten_lim_db} dB" if cfg.atten_lim_db else ""
        self.ctx.log(f"denoised with {denoise.MODEL_NAME}{lim}; converted original kept for A/B listening")

    def step_vad(self, report) -> None:
        src = self._source()
        audio = self._audio_for_clips(src)
        work = self.paths.work_dir(self.source_id)
        work.mkdir(parents=True, exist_ok=True)
        wav16 = work / "vad16k.wav"
        convert.to_wav(audio, wav16, vad.VAD_SR, ffmpeg=self.settings.audio.ffmpeg,
                       duration=src.duration, on_progress=lambda f: report(0.3 * f),
                       check_cancel=self.ctx.check_cancel)
        try:
            probs, engine = vad.speech_probs(wav16, self.settings.vad.engine,
                                             on_progress=lambda f: report(0.3 + 0.7 * f),
                                             check_cancel=self.ctx.check_cancel)
        finally:
            wav16.unlink(missing_ok=True)
        if engine != self.settings.vad.engine:
            self.ctx.warn(f"VAD engine '{self.settings.vad.engine}' unavailable; used '{engine}'")
        np.savez_compressed(work / "vad.npz", probs=probs.astype(np.float16), frame_sec=vad.FRAME_SEC,
                            engine=engine)
        speech = float((probs >= self.settings.vad.threshold).mean()) if len(probs) else 0.0
        self.ctx.log(f"VAD ({engine}): {len(probs)} frames, {speech * 100:.0f}% speech")

    def step_diarize(self, report) -> None:
        src = self._source()
        cfg = self.settings.diarization
        diarizer = diarize.get_diarizer(self.settings)
        if diarizer is None:
            raise RuntimeError("this file is marked multi-speaker but no diarization is available "
                               "(pyannote.audio missing); install it or re-upload as a single-speaker file")
        work = self.paths.work_dir(self.source_id)
        audio = self._audio_for_clips(src)
        wav16 = work / "diar16k.wav"
        try:
            convert.to_wav(audio, wav16, 16000, ffmpeg=self.settings.audio.ffmpeg, duration=src.duration,
                           on_progress=lambda f: report(0.1 * f), check_cancel=self.ctx.check_cancel)
            x16, _ = audio_ops.read_wav(wav16)
        finally:
            wav16.unlink(missing_ok=True)
        data = np.load(work / "vad.npz")
        params = segment.SegParams.from_settings(self.settings)
        regions = [(r.start, r.end) for r in segment.probs_to_regions(data["probs"].astype(np.float32),
                                                                      float(data["frame_sec"]), params)]
        report(0.15, f"{diarizer.name}")
        result = diarizer(x16, regions, max_speakers=cfg.max_speakers)
        del x16
        labels: dict[str, dict] = {}
        for s_, e_, label in result.turns:
            info = labels.setdefault(label, {"seconds": 0.0, "turns": 0, "samples": []})
            info["seconds"] = round(info["seconds"] + (e_ - s_), 2)
            info["turns"] += 1
        samples_dir = work / "speaker_samples"
        shutil.rmtree(samples_dir, ignore_errors=True)
        samples_dir.mkdir(parents=True)
        ccfg = self.settings.clip
        for label, info in labels.items():
            picked: list[tuple[float, float]] = []
            for s_, e_, lab in sorted(result.turns, key=lambda t: t[0] - t[1]):  # longest turns first
                if lab != label or e_ - s_ < 1.5 or any(abs(s_ - p[0]) < 30 for p in picked):
                    continue
                mid = (s_ + e_) / 2
                a, b = max(s_, mid - 3), min(e_, mid + 3)
                x, sr = audio_ops.read_segment(audio, a, b)
                res = audio_ops.process_clip(x, sr, keep_sec=0.1, top_db=ccfg.trim_top_db, target_lufs=ccfg.target_lufs,
                                             peak_dbfs=ccfg.peak_dbfs, max_gain_db=ccfg.max_gain_db)
                out = samples_dir / f"{label}_{len(picked) + 1}.wav"
                audio_ops.write_wav(out, res.audio, sr)
                info["samples"].append(self.paths.rel(out))
                picked.append((s_, e_))
                if len(picked) == 3:
                    break
        payload = {"engine": result.engine, "turns": [list(t) for t in result.turns],
                   "overlaps": [list(o) for o in result.overlaps], "labels": labels}
        (work / "diarization.json").write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        summary = ", ".join(f"{k} {v['seconds'] / 60:.1f} min" for k, v in sorted(labels.items()))
        ovl = sum(e_ - s_ for s_, e_ in result.overlaps)
        self.ctx.log(f"diarization ({result.engine}): {len(labels)} speaker(s): {summary or 'none'}; "
                     f"overlapped speech {ovl:.1f} s")
        # a re-diarized file needs its labels named again
        self._update_source(speaker_map=None)

    def step_assign(self, report) -> None:
        src = self._source()
        labels = list(self._diarization()["labels"])
        mapping = src.get_speaker_map()
        missing = [label for label in labels if label not in mapping]
        if missing:
            raise AwaitingSpeakers(f"{len(labels)} speaker(s) detected, {len(missing)} not named yet")
        names = []
        with session_scope() as s:
            for label in labels:
                m = mapping[label]
                names.append(f"{label} -> ignored" if m.get("ignore") else
                             f"{label} -> {s.get(Speaker, m['speaker_id']).name}")
        self.ctx.log("speakers: " + "; ".join(names))

    def step_clips(self, report) -> None:
        src = self._source()
        audio = self._audio_for_clips(src)
        removed = self._delete_clips()  # a previous partial attempt of this step
        if removed:
            self.ctx.log(f"removed {removed} clips from an interrupted run")
        data = np.load(self.paths.work_dir(self.source_id) / "vad.npz")
        probs, frame_sec = data["probs"].astype(np.float32), float(data["frame_sec"])
        params = segment.SegParams.from_settings(self.settings)
        mapping, overlaps, excluded = {}, [], {}
        if src.multi_speaker:
            diar = self._diarization()
            mapping = src.get_speaker_map()
            keep = {label for label, m in mapping.items() if not m.get("ignore")}
            overlaps = [tuple(o) for o in diar.get("overlaps", [])]
            plan = segment.segment(probs, frame_sec, src.duration or 0.0, params,
                                   turns=[tuple(t) for t in diar["turns"]], keep=keep, stats=excluded)
        else:
            plan = segment.segment(probs, frame_sec, src.duration or 0.0, params)
        self.ctx.log(f"segmentation: {len(plan)} clips planned from the "
                     f"{'enhanced' if src.enhanced_path else 'converted'} audio")
        if excluded:
            self.ctx.log(f"speech kept out of all clips: {excluded['ignored_sec']:.1f} s of ignored speakers, "
                         f"{excluded['unattributed_sec']:.1f} s not attributed to any speaker")
        dcfg = self.settings.diarization

        ccfg, seg = self.settings.clip, self.settings.segmentation
        rows: list[Clip] = []
        stats = {"too_short": 0, "too_long": 0, "forced_cut": 0, "limited": 0}
        for k, pc in enumerate(plan):
            idx = k + 1
            x, sr = audio_ops.read_segment(audio, pc.start, pc.end)
            res = audio_ops.process_clip(x, sr, keep_sec=params.padding, top_db=ccfg.trim_top_db,
                                         target_lufs=ccfg.target_lufs, peak_dbfs=ccfg.peak_dbfs,
                                         max_gain_db=ccfg.max_gain_db)
            out = self.paths.clip_wav(self.source_id, idx)
            audio_ops.write_wav(out, res.audio, sr)
            duration = len(res.audio) / sr
            flags = set(pc.flags)
            status, reject = "pending", None
            if duration < seg.min_duration:
                status, reject = "rejected", "too_short"
            elif duration > seg.hard_max_duration:
                status, reject = "rejected", "too_long"
            if res.lufs_in is not None and res.lufs_in < -50:
                flags.add("very_quiet")
            if overlaps:
                ov = diarize.overlap_seconds(pc.start, pc.end, overlaps)
                if ov > dcfg.overlap_flag_sec:
                    flags.add("overlap")
                    if status == "pending" and ov / max(duration, 1e-6) > dcfg.overlap_reject_ratio:
                        status, reject = "rejected", "overlap"
            if reject:
                stats[reject] = stats.get(reject, 0) + 1
            stats["forced_cut"] += "forced_cut" in flags
            stats["limited"] += res.limited
            speaker_id = mapping[pc.label]["speaker_id"] if src.multi_speaker else src.speaker_id
            clip = Clip(
                source_file_id=self.source_id, speaker_id=speaker_id, speaker_label=pc.label,
                language=src.language, idx=idx,
                clip_path=self.paths.rel(out),
                start_time=round(pc.start + res.trim_start / sr, 3),
                end_time=round(pc.end - res.trim_end / sr, 3),
                duration=round(duration, 3), loudness_lufs=res.lufs_in, status=status, reject_reason=reject,
                needs_review=bool(flags), audio_hash=audio_ops.pcm_hash(res.audio),
            )
            clip.set_flags(flags)
            rows.append(clip)
            if len(rows) >= 50:
                self._insert(rows)
                rows = []
            report(idx / max(len(plan), 1), f"{idx}/{len(plan)} clips")
            self.ctx.check_cancel()
        self._insert(rows)
        self.ctx.log(f"clips written: {len(plan)} (too_short={stats['too_short']}, too_long={stats['too_long']}, "
                     f"forced_cut={stats['forced_cut']}, peak-limited={stats['limited']})")

    @staticmethod
    def _insert(rows: list[Clip]) -> None:
        if rows:
            with session_scope(write=True) as s:
                s.add_all(rows)

    def step_quality(self, report) -> None:
        cfg = self.settings.quality
        scorer = quality.get_dnsmos(self.settings)
        tagger = quality.get_tagger(self.settings)
        status = engines.snapshot()
        self.ctx.log(f"quality: {status.get('quality', 'snr')}; tagger: {status.get('tagger', 'off')}")
        todo = self._clip_rows(Clip.quality_score.is_(None))
        counts: Counter = Counter()
        for n, (clip_id, clip_path, _duration) in enumerate(todo, 1):
            x, sr = audio_ops.read_wav(self.paths.abs(clip_path))
            x16 = audio_ops.resample_poly_to(x, sr, 16000)
            res = scorer.score(x16) if scorer else quality.snr_quality(x, sr)
            info = {"method": res.method, **res.details}
            new_flags: set[str] = set()
            if tagger is not None:
                tags = tagger.tag(x16)
                info.update(tags)
                if tags["music"] >= cfg.tagger.music_threshold:
                    new_flags.add("music")
                if tags["babble"] >= cfg.tagger.babble_threshold:
                    new_flags.add("babble")
            if cfg.reject_below <= res.score < cfg.review_below:
                new_flags.add("borderline_quality")
            with session_scope(write=True) as s:
                clip = s.get(Clip, clip_id)
                clip.quality_score = res.score
                clip.quality_info = json.dumps(info)
                clip.set_flags((set(clip.flags()) - {"borderline_quality", "music", "babble"}) | new_flags)
                if res.score < cfg.reject_below and clip.status == "pending":
                    clip.status, clip.reject_reason = "rejected", "low_quality"
                    counts["low_quality"] += 1
                if clip.status == "pending":
                    clip.needs_review = bool(clip.flags())
            counts.update(new_flags)
            report(n / max(len(todo), 1), f"{n}/{len(todo)} clips")
            self.ctx.check_cancel()
        summary = ", ".join(f"{k}={v}" for k, v in sorted(counts.items())) or "nothing flagged"
        self.ctx.log(f"quality: {len(todo)} clips scored ({summary}); "
                     f"reject < {cfg.reject_below}, review < {cfg.review_below}")

    def step_asr(self, report) -> None:
        src = self._source()
        todo = self._clip_rows(Clip.asr_primary_text.is_(None))
        with session_scope() as s:
            total = s.scalar(select(func.count(Clip.id)).where(Clip.source_file_id == self.source_id)) or 0
        if not todo:
            self.ctx.log("asr: nothing to transcribe")
            return
        engine = asr_mod.get_primary_asr(self.settings)
        if engine is None:
            self.ctx.warn("no primary ASR available: transcripts left empty, clips flagged no_asr")
            with session_scope(write=True) as s:
                for clip_id, _, _ in todo:
                    transcripts.apply_no_primary(s.get(Clip, clip_id))
            return

        lang_cfg = self.settings.languages.get(src.language)
        hint = self.settings.whisper_language(src.language)
        prompt = lang_cfg.initial_prompt if lang_cfg else None
        self.ctx.log(f"asr: {engine.name}, language hint '{hint}', {len(todo)} of {total} clips to transcribe")
        t0, audio_sec, already = time.monotonic(), 0.0, total - len(todo)
        for n, (clip_id, clip_path, duration) in enumerate(todo, 1):
            result = engine.transcribe(self._clip_audio16k(clip_path), hint, prompt)
            with session_scope(write=True) as s:
                transcripts.apply_primary(s.get(Clip, clip_id), result, engine.name, self.settings)
            audio_sec += duration
            speed = audio_sec / max(time.monotonic() - t0, 1e-6)
            report((already + n) / total, f"{already + n}/{total} clips, {speed:.1f}x real-time")
            self.ctx.check_cancel()
        self.ctx.log(f"asr: {len(todo)} clips, {audio_sec / 60:.1f} min audio in {(time.monotonic() - t0) / 60:.1f} min")

    def step_asr2(self, report) -> None:
        src = self._source()
        todo = self._clip_rows(Clip.asr_secondary_text.is_(None), Clip.asr_primary_text.is_not(None))
        if not todo:
            self.ctx.log("asr2: nothing to transcribe")
            return
        engine = asr_mod.get_secondary_asr(self.settings, src.language)
        if engine is None:
            self.ctx.warn("asr2: no secondary ASR available; agreement scores skipped")
            return
        fallback = asr_mod.secondary_fallback_reason(self.settings)
        if fallback and src.language in self.settings.asr.secondary.indicconformer.languages:
            self.ctx.warn(f"asr2: IndicConformer unavailable ({fallback}); using {engine.name}")
        is_indic = isinstance(engine, asr_mod.IndicConformerASR)
        hint = src.language if is_indic else self.settings.whisper_language(src.language)
        self.ctx.log(f"asr2: {engine.name}, language '{hint}', {len(todo)} clips")
        t0, audio_sec = time.monotonic(), 0.0
        for n, (clip_id, clip_path, duration) in enumerate(todo, 1):
            result = engine.transcribe(self._clip_audio16k(clip_path), hint)
            with session_scope(write=True) as s:
                transcripts.apply_secondary(s.get(Clip, clip_id), result.text, engine.name, self.settings)
            audio_sec += duration
            speed = audio_sec / max(time.monotonic() - t0, 1e-6)
            report(n / len(todo), f"{n}/{len(todo)} clips, {speed:.1f}x real-time")
            self.ctx.check_cancel()
        with session_scope() as s:
            mean_wer = s.scalar(select(func.avg(Clip.asr_agreement)).where(Clip.source_file_id == self.source_id))
        mean_txt = "n/a" if mean_wer is None else f"{mean_wer:.3f}"
        self.ctx.log(f"asr2: {len(todo)} clips in {(time.monotonic() - t0) / 60:.1f} min, "
                     f"mean primary/secondary WER {mean_txt}")

    def step_speakers(self, report) -> None:
        cfg = self.settings.speaker_check
        if not cfg.enabled:
            self.ctx.log("speaker check: disabled in config")
            return
        embedder = speakers_mod.get_embedder(self.settings)
        if embedder is None:
            self.ctx.warn(f"speaker check skipped: {engines.snapshot().get('speaker_embedding', 'unavailable')}")
            return
        todo = self._clip_rows(Clip.speaker_embedding.is_(None))
        for n, (clip_id, clip_path, _duration) in enumerate(todo, 1):
            emb = embedder.embed(self._clip_audio16k(clip_path))
            with session_scope(write=True) as s:
                s.get(Clip, clip_id).speaker_embedding = speakers_mod.to_blob(emb)
            report(0.9 * n / max(len(todo), 1), f"{n}/{len(todo)} embeddings")
            self.ctx.check_cancel()
        with session_scope() as s:
            speaker_ids = sorted(set(s.scalars(select(Clip.speaker_id).where(Clip.source_file_id == self.source_id))))
            names = {sp.id: sp.name for sp in s.scalars(select(Speaker).where(Speaker.id.in_(speaker_ids)))}
        for speaker_id in speaker_ids:
            st = speakers_mod.check_speaker(speaker_id, self.settings)
            if st.get("skipped"):
                self.ctx.log(f"speaker check {names.get(speaker_id)}: skipped ({st['clips']} clips < {cfg.min_clips})")
            else:
                self.ctx.log(f"speaker check {names.get(speaker_id)}: {st['clips']} clips, median similarity "
                             f"{st['median_similarity']}, {st['flagged']} flagged speaker_mismatch")
        report(1.0)

    def step_dedup(self, report) -> None:
        cfg = self.settings.dedup
        if not cfg.enabled:
            self.ctx.log("dedup: disabled in config")
            return
        pairs = dedup.find_duplicates(self.source_id, self.paths, cfg.min_similarity)
        stats = dedup.apply_duplicates(pairs)
        self.ctx.log(f"dedup: {stats['exact']} exact + {stats['near']} near-identical duplicates "
                     f"({stats['rejected']} pending clips rejected as duplicate)")
        report(1.0)

    def step_finalize(self, report) -> None:
        with session_scope() as s:
            rows = s.execute(
                select(Clip.status, func.count(Clip.id), func.sum(Clip.duration))
                .where(Clip.source_file_id == self.source_id).group_by(Clip.status)
            ).all()
            review = s.scalar(select(func.count(Clip.id)).where(
                Clip.source_file_id == self.source_id, Clip.needs_review.is_(True), Clip.status == "pending")) or 0
            avg_q = s.scalar(select(func.avg(Clip.quality_score)).where(Clip.source_file_id == self.source_id))
        summary = ", ".join(f"{st}={n} ({(d or 0) / 60:.1f} min)" for st, n, d in rows) or "no clips"
        quality_txt = f"; mean quality {avg_q:.2f}" if avg_q is not None else ""
        self.ctx.log(f"summary: {summary}; needs review: {review}{quality_txt}")
        report(1.0)
        self._update_source(updated_at=utcnow())


# ---------------------------------------------------------------- whole-database jobs
def run_speaker_check(ctx: JobContext, settings: Settings) -> None:
    """Recompute embeddings that are missing and re-check every speaker (centroids move as data grows)."""
    embedder = speakers_mod.get_embedder(settings)
    if embedder is None:
        raise RuntimeError(f"speaker embeddings unavailable: {engines.snapshot().get('speaker_embedding')}")
    paths = DataPaths.from_settings(settings)
    with session_scope() as s:
        todo = s.execute(select(Clip.id, Clip.clip_path).where(Clip.speaker_embedding.is_(None))).all()
        speaker_ids = sorted(set(s.scalars(select(Clip.speaker_id))))
        names = {sp.id: sp.name for sp in s.scalars(select(Speaker))}
    for n, (clip_id, clip_path) in enumerate(todo, 1):
        x, sr = audio_ops.read_wav(paths.abs(clip_path))
        emb = embedder.embed(audio_ops.resample_poly_to(x, sr, 16000))
        with session_scope(write=True) as s:
            s.get(Clip, clip_id).speaker_embedding = speakers_mod.to_blob(emb)
        ctx.progress(0.8 * n / max(len(todo), 1), f"embeddings {n}/{len(todo)}")
        ctx.check_cancel()
    for k, speaker_id in enumerate(speaker_ids, 1):
        st = speakers_mod.check_speaker(speaker_id, settings)
        ctx.log(f"{names.get(speaker_id)}: {st['clips']} clips, "
                + (f"skipped ({st['skipped']})" if st.get("skipped") else
                   f"median similarity {st['median_similarity']}, {st['flagged']} speaker_mismatch"))
        ctx.progress(0.8 + 0.2 * k / len(speaker_ids), f"speaker {k}/{len(speaker_ids)}")


def run_dedup_all(ctx: JobContext, settings: Settings) -> None:
    paths = DataPaths.from_settings(settings)
    with session_scope() as s:
        source_ids = list(s.scalars(select(SourceFile.id).order_by(SourceFile.id)))
    totals = Counter()
    for k, source_id in enumerate(source_ids, 1):
        totals.update(dedup.apply_duplicates(dedup.find_duplicates(source_id, paths, settings.dedup.min_similarity)))
        ctx.progress(k / max(len(source_ids), 1), f"file {k}/{len(source_ids)}")
        ctx.check_cancel()
    ctx.log(f"dedup: {totals['exact']} exact + {totals['near']} near-identical duplicates, "
            f"{totals['rejected']} pending clips rejected")
