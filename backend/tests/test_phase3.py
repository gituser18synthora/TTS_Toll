"""Phase 3: diarization flow, speaker consistency, dedup, pretrain-check, dump-numbers, migration."""
from __future__ import annotations

import io
import itertools
import json
import shutil
import sqlite3
import subprocess

import numpy as np
import pytest
import soundfile as sf

from app.pipeline.asr import AsrResult
from app.pipeline.dedup import same_recording_score
from app.pipeline.diarize import ClusteringDiarizer, Diarization, merge_turns
from app.pipeline.segment import SegParams, segment
from app.pipeline.speakers import mismatch_mask, robust_centroid

from .conftest import make_recording, speechlike
from .test_pipeline_export import all_clips, run_all_jobs, upload
from .test_segment import FRAME, probs_from_spans

_seed = itertools.count(500)


class FakeASR:
    def __init__(self, text):
        self.name, self.text = "fake", text

    def transcribe(self, audio, language, initial_prompt=None):
        return AsrResult(text=self.text, avg_logprob=-0.1, no_speech_prob=0.01, compression_ratio=1.0)


@pytest.fixture
def fixed_asr(monkeypatch):
    from app.pipeline import asr

    monkeypatch.setattr(asr, "get_primary_asr", lambda settings: FakeASR("एक दो तीन चार पाँच"))
    monkeypatch.setattr(asr, "get_secondary_asr", lambda settings, language: None)


# ------------------------------------------------------------------ diarization
def two_speaker_layout():
    """A and B alternate; returns layout + list of (start, end, speaker) speech spans."""
    layout, spans, t = [("silence", 0.5)], [], 0.5
    for k, dur in enumerate([3.0, 2.5, 4.0, 3.5, 2.8, 4.2, 3.1, 2.6]):
        layout.append(("speech", dur))
        spans.append((t, t + dur, "A" if k % 2 == 0 else "B"))
        t += dur
        layout.append(("silence", 0.6))
        t += 0.6
    return layout, spans


class FakeDiarizer:
    name = "fake diarizer"

    def __init__(self, spans, overlaps):
        self.spans, self.overlaps = spans, overlaps

    def __call__(self, audio16k, regions, num_speakers=None, max_speakers=None):
        turns = [(s, e, "SPEAKER_00" if who == "A" else "SPEAKER_01") for s, e, who in self.spans]
        return Diarization(turns, self.overlaps, self.name)


def test_multi_speaker_upload_diarize_assign_and_rename(client, tmp_path, monkeypatch, fixed_asr):
    from app.pipeline import diarize

    layout, spans = two_speaker_layout()
    wav = tmp_path / "dialogue.wav"
    make_recording(wav, layout, seed=next(_seed))
    a_spans = [(s, e) for s, e, who in spans if who == "A"]
    overlap = (a_spans[2][0] + 0.5, a_spans[2][0] + 2.0)  # 1.5 s of crosstalk inside A's third turn
    monkeypatch.setattr(diarize, "get_diarizer", lambda settings: FakeDiarizer(spans, [overlap]))

    r = client.post("/api/uploads", params={"filename": wav.name, "language": "hi", "multi_speaker": "true"},
                    content=wav.read_bytes())
    assert r.status_code == 200, r.text
    sid = r.json()["id"]
    assert r.json()["multi_speaker"] and r.json()["speaker_id"] is None
    run_all_jobs()

    src = client.get(f"/api/sources/{sid}").json()
    assert src["status"] == "needs_speakers"
    assert "diarize" in src["steps_done"] and "clips" not in src["steps_done"]
    view = client.get(f"/api/sources/{sid}/speakers").json()
    assert [l["label"] for l in view["labels"]] == ["SPEAKER_00", "SPEAKER_01"]
    sample = view["labels"][0]["samples"][0]
    audio = client.get(sample)
    assert audio.status_code == 200 and sf.info(io.BytesIO(audio.content)).samplerate == 24000

    bad = client.post(f"/api/sources/{sid}/speakers", json={"assignments": [{"label": "SPEAKER_00", "name": "x"}]})
    assert bad.status_code == 422  # gender missing
    r = client.post(f"/api/sources/{sid}/speakers", json={"assignments": [
        {"label": "SPEAKER_00", "name": "alice", "gender": "female"},
        {"label": "SPEAKER_01", "ignore": True}]})
    assert r.status_code == 200, r.text
    run_all_jobs()

    src = client.get(f"/api/sources/{sid}").json()
    assert src["status"] == "done" and src["speaker_name"] == "alice"
    clips = all_clips(client, sid)
    assert clips and {c["speaker_name"] for c in clips} == {"alice"} and {c["speaker_label"] for c in clips} == {"SPEAKER_00"}
    for c in clips:  # every clip lies inside one of A's turns (plus padding) - never in B's speech
        assert any(s - 0.31 <= c["start_time"] and c["end_time"] <= e + 0.31 for s, e in a_spans), c
    hit = [c for c in clips if c["start_time"] < overlap[1] and c["end_time"] > overlap[0]]
    assert hit and all("overlap" in c["flags"] for c in hit)
    assert all(c["status"] == "rejected" and c["reject_reason"] == "overlap" for c in hit)  # 1.5 s of a ~4 s clip

    # rename after clips exist: clips follow; switching "ignore" needs Reprocess
    r = client.post(f"/api/sources/{sid}/speakers", json={"assignments": [
        {"label": "SPEAKER_00", "name": "alicia", "gender": "female"}]})
    assert r.status_code == 200
    assert {c["speaker_name"] for c in all_clips(client, sid)} == {"alicia"}
    r = client.post(f"/api/sources/{sid}/speakers", json={"assignments": [
        {"label": "SPEAKER_01", "name": "bob", "gender": "male"}]})
    assert r.status_code == 409
    assert client.patch(f"/api/sources/{sid}", json={"speaker_name": "zed"}).status_code == 409
    run_all_jobs()  # the speaker_check job queued by the rename (disabled in tests -> no-op)


def test_single_speaker_upload_still_requires_speaker(client, tmp_path):
    wav = tmp_path / "x.wav"
    make_recording(wav, [("speech", 2.0)], seed=next(_seed))
    r = client.post("/api/uploads", params={"filename": wav.name, "language": "hi"}, content=wav.read_bytes())
    assert r.status_code == 422


def test_merge_turns_joins_and_absorbs_blips():
    turns = [(0.0, 2.0, "a"), (2.1, 3.0, "a"), (3.0, 3.1, "b"), (3.1, 5.0, "a"), (5.5, 7.0, "b")]
    assert merge_turns(turns) == [(0.0, 5.0, "a"), (5.5, 7.0, "b")]
    # a short turn between pauses is nobody's - never handed to the previous speaker across the pause
    # (real case: the other narrator's first syllable ended up in the previous narrator's clip)
    assert merge_turns([(0.0, 2.0, "a"), (2.7, 2.9, "a"), (3.6, 5.0, "b")]) == [(0.0, 2.0, "a"), (3.6, 5.0, "b")]
    assert merge_turns([(0.0, 2.0, "a"), (2.7, 2.9, "b"), (3.6, 5.0, "b")]) == [(0.0, 2.0, "a"), (3.6, 5.0, "b")]


def test_clips_never_span_or_pad_into_excluded_speech():
    """An ignored speaker's "haan" and an unattributed blip, each within merge distance of the kept
    speaker's speech, split that speaker's clips instead of ending up inside one."""
    spans = [(0.5, 3.5), (3.8, 4.1), (4.4, 7.4), (7.7, 7.9), (8.2, 11.0)]
    turns = [(0.5, 3.5, "A"), (3.8, 4.1, "B"), (4.4, 7.4, "A"), (8.35, 11.0, "A")]  # 7.7-7.9: nobody's
    stats: dict = {}
    clips = segment(probs_from_spans(spans, 12.0), FRAME, 12.0, SegParams(), turns=turns, keep={"A"}, stats=stats)
    assert [c.label for c in clips] == ["A", "A", "A"]
    for c in clips:
        for a, b in ((3.8, 4.1), (7.7, 7.9)):
            assert c.end <= a or c.start >= b, f"clip {c.start:.2f}-{c.end:.2f} contains excluded {a}-{b}"
    assert clips[2].speech_start == pytest.approx(8.2, abs=0.02)  # 0.15 s VAD/diarizer jitter is not cut off
    assert stats["ignored_sec"] == pytest.approx(0.3, abs=0.03)
    assert stats["unattributed_sec"] == pytest.approx(0.2, abs=0.03)


class SignEmbedder:
    """Fake speaker embeddings: speaker A is recorded as a positive DC level, B as a negative one."""

    def embed(self, x):
        v = np.array([1.0, 0.05]) if float(np.mean(x)) > 0 else np.array([0.05, 1.0])
        return v / np.linalg.norm(v)


def test_clustering_diarizer_attributes_short_regions_only_by_contact():
    sr = 16000
    layout = [(0.0, 3.0, 1), (3.6, 3.85, -1), (4.5, 7.5, -1), (7.7, 7.95, -1)]  # A, lone B blip, B, touching B
    audio = np.zeros(9 * sr, dtype=np.float32)
    for s, e, sign in layout:
        audio[int(s * sr):int(e * sr)] = 0.1 * sign
    d = ClusteringDiarizer(SignEmbedder(), threshold=0.7)(audio, [(s, e) for s, e, _ in layout])
    assert [(round(s, 2), round(e, 2)) for s, e, _ in d.turns] == [(0.0, 3.0), (4.5, 7.95)]
    assert d.turns[0][2] != d.turns[1][2]


# ------------------------------------------------------------------ speaker consistency
def test_robust_centroid_and_mismatch_mask():
    rng = np.random.default_rng(0)
    base = rng.standard_normal(256)
    same = np.stack([base + 0.4 * rng.standard_normal(256) for _ in range(40)])
    other = rng.standard_normal((2, 256))
    embs = np.concatenate([same, other])
    embs /= np.linalg.norm(embs, axis=1, keepdims=True)
    sims = embs @ robust_centroid(embs)
    mask = mismatch_mask(sims, min_similarity=0.5)
    assert mask[-2:].all() and not mask[:-2].any()


def test_speaker_check_flags_the_odd_clip(client, tmp_path, monkeypatch, fixed_asr):
    from app.config import get_settings
    from app.pipeline import speakers

    class FakeEmbedder:
        """Same 'voice' for every clip except the longest one."""
        def __init__(self):
            self.calls = []

        def embed(self, audio16k):
            self.calls.append(len(audio16k))
            v = np.zeros(256, dtype=np.float32)
            v[0 if len(audio16k) < 16000 * 4.6 else 1] = 1.0
            return v

    fake = FakeEmbedder()
    monkeypatch.setattr(speakers, "get_embedder", lambda settings: fake)
    cfg = get_settings().speaker_check
    monkeypatch.setattr(cfg, "enabled", True)
    layout = [("silence", 0.5)]
    for dur in (2.5, 3.0, 2.8, 3.2, 2.6, 5.0, 3.1):  # the 5 s clip is the odd voice
        layout += [("speech", dur), ("silence", 2.0)]  # > max_pause_in_clip: one utterance per clip
    wav = tmp_path / "voices.wav"
    make_recording(wav, layout, seed=next(_seed))
    sid = upload(client, wav, speaker="voicecheck", gender="male", language="hi").json()["id"]
    run_all_jobs()
    clips = [c for c in all_clips(client, sid) if c["status"] == "pending"]
    odd = [c for c in clips if "speaker_mismatch" in c["flags"]]
    assert len(odd) == 1 and odd[0]["duration"] > 4.6 and odd[0]["needs_review"]
    assert all(c["speaker_similarity"] is not None for c in clips)


# ------------------------------------------------------------------ dedup
def test_same_recording_score_separates_copies_from_new_takes():
    rng = np.random.default_rng(3)
    a = speechlike(4.0, 24000, rng)
    copy = np.concatenate([np.zeros(2400, dtype=np.float32), 0.6 * a])[: len(a)]  # re-trimmed + quieter
    retake = speechlike(4.0, 24000, np.random.default_rng(99))                       # same 'words', new take
    assert same_recording_score(a, copy, 24000) > 0.9
    assert same_recording_score(a, retake, 24000) < 0.3


def test_dedup_exact_and_near_identical(client, tmp_path, fixed_asr):
    layout = [("silence", 0.5)]
    for dur in (2.6, 3.4, 2.9, 3.8):
        layout += [("speech", dur), ("silence", 0.9)]
    wav = tmp_path / "orig.wav"
    make_recording(wav, layout, seed=next(_seed))
    first = upload(client, wav, speaker="dupspk", gender="female", language="hi").json()["id"]
    # exact: the same PCM in a FLAC container (different file hash, identical audio)
    flac = tmp_path / "orig_copy.flac"
    x, sr = sf.read(str(wav), dtype="int16")
    sf.write(str(flac), x, sr, subtype="PCM_16")
    exact = upload(client, flac, speaker="dupspk", gender="female", language="hi").json()["id"]
    # near: re-encoded copy (resampled + quieter + tiny noise floor)
    near = tmp_path / "orig_reencoded.wav"
    subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-i", str(wav), "-ar", "22050",
                    "-af", "volume=0.7", str(near)], check=True)
    near_id = upload(client, near, speaker="dupspk", gender="female", language="hi").json()["id"]
    run_all_jobs()

    kept = all_clips(client, first)
    assert kept and all("duplicate" not in c["flags"] for c in kept)
    kept_ids = {c["id"] for c in kept}
    for sid, kind in ((exact, "exact"), (near_id, "near")):
        clips = all_clips(client, sid)
        assert clips, kind
        for c in clips:
            assert c["status"] == "rejected" and c["reject_reason"] == "duplicate", (kind, c)
            assert c["duplicate_of"] in kept_ids
    # a global re-run changes nothing
    job = client.post("/api/maintenance/dedup").json()
    run_all_jobs()
    assert client.get(f"/api/jobs/{job['id']}").json()["status"] == "done"
    assert all("duplicate" not in c["flags"] for c in all_clips(client, first))


# ------------------------------------------------------------------ pretrain-check
@pytest.fixture
def good_export(client, portal_env, tmp_path, fixed_asr):
    layout = [("silence", 0.5)]
    for dur in (2.7, 3.3, 4.1, 2.9):
        layout += [("speech", dur), ("silence", 0.8)]
    wav = tmp_path / "pc.wav"
    make_recording(wav, layout, seed=next(_seed))
    sid = upload(client, wav, speaker="pcspk", gender="male", language="hi").json()["id"]
    run_all_jobs()
    ids = [c["id"] for c in all_clips(client, sid) if c["status"] == "pending"]
    client.post("/api/clips/bulk", json={"ids": ids, "action": "approve"})
    speaker_id = all_clips(client, sid)[0]["speaker_id"]
    exp = client.post("/api/exports", json={"name": f"pc_{sid}", "speaker_ids": [speaker_id],
                                            "formats": ["jsonl"], "script": "both"}).json()
    run_all_jobs()
    exp = client.get(f"/api/exports/{exp['id']}").json()
    assert exp["status"] == "done", exp["error"]
    return portal_env / "data" / exp["out_dir"]


def test_pretrain_check_passes_clean_export(good_export):
    from app.export.pretrain_check import check_export

    rep = check_export(good_export, min_duration=2.0, max_duration=20.0)
    assert not rep.errors, [i.__dict__ for i in rep.errors]
    assert rep.speakers["pcspk"]["rows"] == 2 * rep.speakers["pcspk"]["clips"]  # script: both
    assert [i.code for i in rep.warnings] == ["low_speaker_data"]


def test_pretrain_check_catches_problems(good_export, tmp_path):
    from app.export.pretrain_check import check_export, format_report

    bad = tmp_path / "bad_export"
    shutil.copytree(good_export, bad)
    rows = [json.loads(x) for x in (bad / "train.jsonl").read_text(encoding="utf-8").splitlines()]
    manifest = [json.loads(x) for x in (bad / "manifest.jsonl").read_text(encoding="utf-8").splitlines()]
    rows[0]["text"] = rows[0]["text"].split(": ", 1)[1]            # missing speaker prefix
    rows[1]["text"] = "pcspk: "                                     # empty transcript
    rows[2]["text"] = "pcspk: இது தமிழ் உரை"                          # Tamil text in a Hindi row
    rows.append(dict(rows[3])); manifest.append(dict(manifest[3]))  # duplicate row
    manifest[3]["needs_review"] = True                              # exported while flagged
    wav16 = bad / rows[4]["audio"]
    x, _ = sf.read(str(wav16), dtype="float32")
    sf.write(str(wav16), x[::3], 8000, subtype="PCM_16")            # wrong format (and checksum mismatch)
    (bad / "train.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    (bad / "manifest.jsonl").write_text("".join(json.dumps(m, ensure_ascii=False) + "\n" for m in manifest), encoding="utf-8")

    rep = check_export(bad, min_duration=2.0, max_duration=20.0)
    codes = {i.code for i in rep.errors}
    assert {"missing_prefix", "empty_text", "wrong_script", "duplicate_row", "needs_review", "bad_format",
            "checksum_mismatch"} <= codes, codes
    assert "RESULT: FAIL" in format_report(rep)


def test_pretrain_check_cli_exit_codes(good_export, tmp_path, monkeypatch):
    import sys

    from app import cli

    monkeypatch.setattr(sys, "argv", ["cli", "pretrain-check", str(good_export), "--json", str(tmp_path / "r.json")])
    with pytest.raises(SystemExit) as ok:
        cli.main()
    assert ok.value.code == 0 and json.loads((tmp_path / "r.json").read_text())["ok"] is True
    broken = tmp_path / "broken"
    shutil.copytree(good_export, broken)
    (broken / "train.jsonl").write_text('{"audio": "wavs/missing.wav", "text": "pcspk: x"}\n', encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["cli", "pretrain-check", str(broken), "--no-checksums"])
    with pytest.raises(SystemExit) as fail:
        cli.main()
    assert fail.value.code == 1


# ------------------------------------------------------------------ dump-numbers
def test_dump_numbers_writes_review_sheet(tmp_path, monkeypatch, portal_env):
    import sys

    from app import cli

    out = tmp_path / "numbers_hi.tsv"
    monkeypatch.setattr(sys, "argv", ["cli", "dump-numbers", "--lang", "hi", "--out", str(out)])
    cli.main()
    lines = out.read_text(encoding="utf-8").splitlines()
    header = lines[0].split("\t")
    assert header[3].startswith("spoken") and header[4].startswith("roman") and len(lines) == 47
    rows = {r.split("\t")[2]: r.split("\t") for r in lines[1:]}
    assert rows["₹500"][3] == "पाँच सौ रुपये" and rows["1,00,000"][3] == "एक लाख" and rows["2.5"][4] == "do dashamlav paanch"


# ------------------------------------------------------------------ migration
def test_not_null_relaxation_migration(tmp_path):
    from sqlalchemy import create_engine

    from app import db as dbmod
    from app.models import Base

    path = tmp_path / "old.db"
    engine = create_engine(f"sqlite:///{path}")
    Base.metadata.create_all(engine)
    engine.dispose()
    con = sqlite3.connect(path)  # simulate the phase 1/2 schema: source_files.speaker_id NOT NULL
    ddl = con.execute("select sql from sqlite_master where name='source_files'").fetchone()[0]
    con.executescript(f"""
        PRAGMA foreign_keys=OFF;
        DROP TABLE source_files;
        {ddl.replace('speaker_id INTEGER,', 'speaker_id INTEGER NOT NULL,')};
        INSERT INTO speakers (id, name, gender, created_at) VALUES (1, 'old', 'male', '2026-01-01');
        INSERT INTO source_files (id, original_path, original_filename, speaker_id, language, status, steps_done,
                                  progress, multi_speaker, created_at, updated_at)
        VALUES (7, 'raw/x.wav', 'x.wav', 1, 'hi', 'done', '[]', 100, 0, '2026-01-01', '2026-01-01');
    """)
    assert [r[3] for r in con.execute("PRAGMA table_info(source_files)") if r[1] == "speaker_id"] == [1]
    con.close()
    try:
        dbmod.init_engine(path)
        dbmod.create_schema()
        con = sqlite3.connect(path)
        assert [r[3] for r in con.execute("PRAGMA table_info(source_files)") if r[1] == "speaker_id"] == [0]
        assert con.execute("select id, original_filename, speaker_id from source_files").fetchall() == [(7, "x.wav", 1)]
        assert con.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        con.close()
    finally:  # restore the session-wide test database
        from app.config import get_settings
        from app.paths import DataPaths

        dbmod.init_engine(DataPaths.from_settings(get_settings()).db_file)
