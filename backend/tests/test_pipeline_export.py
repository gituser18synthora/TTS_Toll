"""End-to-end: upload -> process -> review -> export, plus reprocess and resume."""
from __future__ import annotations

import hashlib
import json
import zipfile

import numpy as np
import pytest
import soundfile as sf

from .conftest import make_recording


def run_all_jobs() -> list[int]:
    from app.config import get_settings
    from app.worker.jobs import claim_next
    from app.worker.runner import run_job

    done = []
    while (job_id := claim_next("pytest:0")) is not None:
        run_job(job_id, get_settings())
        done.append(job_id)
    return done


def sentence_layout(seed: int, n: int = 14) -> list[tuple[str, float]]:
    rng = np.random.default_rng(seed)
    layout = [("silence", 0.8)]
    for i in range(n):
        layout.append(("speech", float(rng.uniform(1.5, 5.0))))
        layout.append(("silence", float(rng.choice([0.3, 0.5, 0.8]))))
    layout += [("silence", 2.5), ("speech", 0.6), ("silence", 2.5)]  # isolated short utterance -> too_short
    return layout


def upload(client, path, speaker="tester", gender="female", language="hi", **extra):
    return client.post("/api/uploads", params={"filename": path.name, "speaker_name": speaker, "gender": gender,
                                               "language": language, **extra}, content=path.read_bytes())


@pytest.fixture(scope="module")
def processed(client, portal_env, tmp_path_factory):
    wav = tmp_path_factory.mktemp("rec") / "session1.wav"
    spans = make_recording(wav, sentence_layout(0))
    r = upload(client, wav)
    assert r.status_code == 200, r.text
    run_all_jobs()
    return r.json()["id"], spans


def all_clips(client, source_id):
    return client.get("/api/clips", params={"source_file_id": source_id, "status": "all", "page_size": 500}).json()["items"]


def test_processing_produces_valid_clips(client, portal_env, processed):
    source_id, spans = processed
    src = client.get(f"/api/sources/{source_id}").json()
    assert src["status"] == "done", src["error"]
    assert src["steps_done"] == ["convert", "denoise", "vad", "diarize", "assign", "clips", "quality", "asr", "asr2",
                                 "speakers", "dedup", "finalize"]
    data = portal_env / "data"
    assert (data / src["original_path"]).exists()

    clips = all_clips(client, source_id)
    assert clips
    for c in clips:
        info = sf.info(str(data / "clips" / f"src_{source_id:05d}" / f"src_{source_id:05d}_{c['idx']:05d}.wav"))
        assert (info.samplerate, info.channels, info.subtype) == (24000, 1, "PCM_16")
        assert abs(info.frames / 24000 - c["duration"]) < 0.01
        if c["status"] == "pending":
            assert 2.0 <= c["duration"] <= 20.0
        else:
            assert c["reject_reason"] == "too_short" and c["duration"] < 2.0
        assert "no_asr" in c["flags"]  # test config runs without ASR

    # every speech span lies completely inside one clip: nothing was cut mid-utterance
    for a, b in spans:
        holders = [c for c in clips if c["start_time"] - 0.06 <= a and b <= c["end_time"] + 0.06]
        assert len(holders) == 1, f"speech {a:.2f}-{b:.2f} not inside exactly one clip"


def test_review_and_export_jsonl(client, portal_env, processed):
    source_id, _ = processed
    clips = [c for c in all_clips(client, source_id) if c["status"] == "pending"]
    r = client.patch(f"/api/clips/{clips[0]['id']}", json={"status": "approved"})
    assert r.status_code == 422  # empty transcript cannot be approved

    for c in clips:
        r = client.patch(f"/api/clips/{c['id']}", json={"text": f"  वाक्य  नंबर {c['idx']} ,ठीक है ", "status": "approved"})
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["status"] == "approved" and body["text"] == f"वाक्य नंबर {c['idx']}, ठीक है"
        assert body["text_edited"] and not body["needs_review"] and body["reviewed_at"]

    speaker_id = clips[0]["speaker_id"]  # other test modules leave approved clips of other speakers
    preview = client.get("/api/exports/preview", params={"speaker_ids": [speaker_id]}).json()
    assert preview["clips"] == len(clips)

    r = client.post("/api/exports", json={"name": "test export", "formats": ["jsonl", "csv"],
                                          "speaker_ids": [speaker_id]})
    assert r.status_code == 200, r.text
    export_id = r.json()["id"]
    run_all_jobs()
    exp = client.get(f"/api/exports/{export_id}").json()
    assert exp["status"] == "done", exp["error"]
    assert exp["clip_count"] == len(clips)

    out = portal_env / "data" / exp["out_dir"]
    lines = (out / "train.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == len(clips)
    for line in lines:
        row = json.loads(line)
        assert set(row) == {"audio", "text"}
        assert row["text"].startswith("tester: वाक्य नंबर ")
        assert row["audio"].startswith("wavs/") and not row["audio"].startswith("/")
        info = sf.info(str(out / row["audio"]))
        assert (info.samplerate, info.channels, info.subtype) == (24000, 1, "PCM_16")

    manifest = [json.loads(x) for x in (out / "manifest.jsonl").read_text(encoding="utf-8").splitlines()]
    assert {m["speaker"] for m in manifest} == {"tester"} and {m["gender"] for m in manifest} == {"female"}
    for row in (out / "metadata.csv").read_text(encoding="utf-8").splitlines():
        uid, text, norm, speaker, lang, script = row.split("|")
        assert (out / "wavs" / f"{uid}.wav").exists() and speaker == "tester" and lang == "hi" and text == norm
        assert script == "devanagari"

    for row in (out / "SHA256SUMS").read_text().splitlines():
        digest, rel = row.split("  ", 1)
        assert hashlib.sha256((out / rel).read_bytes()).hexdigest() == digest

    stats = json.loads((out / "stats.json").read_text())
    assert stats["clips"] == len(clips) and "tester" in stats["per_speaker"] and "hi" in stats["per_language"]

    with zipfile.ZipFile(portal_env / "data" / exp["zip_path"]) as zf:
        names = set(zf.namelist())
    assert {"test_export/train.jsonl", "test_export/SHA256SUMS", "test_export/export.log"} <= names
    assert sum(n.endswith(".wav") for n in names) == len(clips)

    # re-running rebuilds the same export from the currently approved clips
    client.patch(f"/api/clips/{clips[0]['id']}", json={"status": "rejected", "reject_reason": "noise"})
    assert client.post(f"/api/exports/{export_id}/rerun").status_code == 200
    run_all_jobs()
    exp = client.get(f"/api/exports/{export_id}").json()
    assert exp["status"] == "done" and exp["clip_count"] == len(clips) - 1
    assert len((out / "train.jsonl").read_text(encoding="utf-8").splitlines()) == len(clips) - 1


def test_bulk_actions(client, processed):
    source_id, _ = processed
    ids = [c["id"] for c in all_clips(client, source_id)][:3]
    r = client.post("/api/clips/bulk", json={"ids": ids, "action": "pending"})
    assert r.json()["updated"] == 3
    r = client.post("/api/clips/bulk", json={"ids": ids, "action": "reject", "reject_reason": "music"})
    assert r.json()["updated"] == 3
    assert {c["reject_reason"] for c in all_clips(client, source_id) if c["id"] in ids} == {"music"}


def test_reprocess_replaces_clips_and_keeps_original(client, portal_env, tmp_path):
    wav = tmp_path / "session2.wav"
    make_recording(wav, sentence_layout(1, n=6), seed=1)
    sid = upload(client, wav, speaker="second", gender="male", language="en").json()["id"]
    run_all_jobs()
    before = {c["id"] for c in all_clips(client, sid)}
    original = portal_env / "data" / client.get(f"/api/sources/{sid}").json()["original_path"]
    digest = hashlib.sha256(original.read_bytes()).hexdigest()

    assert client.post(f"/api/sources/{sid}/reprocess").status_code == 200
    assert client.post(f"/api/sources/{sid}/reprocess").status_code == 409  # already queued
    run_all_jobs()
    after = {c["id"] for c in all_clips(client, sid)}
    assert after and not (before & after) and len(after) == len(before)
    assert hashlib.sha256(original.read_bytes()).hexdigest() == digest


def test_failed_job_resumes_from_last_step(client, tmp_path, monkeypatch):
    from app.pipeline.orchestrator import SourcePipeline

    wav = tmp_path / "session3.wav"
    make_recording(wav, sentence_layout(2, n=5), seed=2)
    original_finalize = SourcePipeline.step_finalize

    def boom(self, report):
        raise RuntimeError("simulated crash")

    monkeypatch.setattr(SourcePipeline, "step_finalize", boom)
    sid = upload(client, wav, speaker="third", gender="male", language="ta").json()["id"]
    run_all_jobs()
    src = client.get(f"/api/sources/{sid}").json()
    assert src["status"] == "failed" and "simulated crash" in src["error"]
    assert src["steps_done"] == ["convert", "denoise", "vad", "diarize", "assign", "clips", "quality", "asr", "asr2",
                                 "speakers", "dedup"]
    n_clips = len(all_clips(client, sid))

    monkeypatch.setattr(SourcePipeline, "step_finalize", original_finalize)
    assert client.post(f"/api/sources/{sid}/process").status_code == 200
    run_all_jobs()
    src = client.get(f"/api/sources/{sid}").json()
    assert src["status"] == "done"
    assert len(all_clips(client, sid)) == n_clips
    log = client.get(f"/api/jobs/{src['job']['id']}").json()["log"]
    assert "step clips: already done, skipping" in log


def test_upload_validation(client, tmp_path):
    wav = tmp_path / "v.wav"
    make_recording(wav, [("speech", 2.0)], seed=9)
    assert upload(client, wav, speaker="bad:name").status_code == 422
    assert upload(client, wav, language="xx").status_code == 422
    assert upload(client, wav, speaker="tester", gender="male").status_code == 409  # tester is female
    bad = tmp_path / "notes.txt"
    bad.write_text("hello")
    assert upload(client, bad).status_code == 415
    fake = tmp_path / "fake.wav"
    fake.write_bytes(b"definitely not audio")
    assert upload(client, fake).status_code == 400
    ok = upload(client, wav, speaker="dup", auto_process="false")
    assert ok.status_code == 200 and ok.json()["job"] is None
    assert upload(client, wav, speaker="dup").status_code == 409  # identical file
