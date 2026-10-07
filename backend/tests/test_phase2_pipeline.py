"""Dual ASR + agreement, re-transcribe preservation rules, script/HF export, quality auto-reject."""
from __future__ import annotations

import io
import itertools
import json

import pytest
import soundfile as sf

from app.pipeline.asr import AsrResult

from .conftest import make_recording
from .test_pipeline_export import all_clips, run_all_jobs, sentence_layout, upload


class FakeASR:
    def __init__(self, name: str, text: str, fail_after: int | None = None):
        self.name, self.text, self.fail_after, self.calls = name, text, fail_after, 0

    def transcribe(self, audio, language, initial_prompt=None) -> AsrResult:
        self.calls += 1
        if self.fail_after is not None and self.calls > self.fail_after:
            raise RuntimeError("simulated ASR crash")
        return AsrResult(text=self.text, avg_logprob=-0.2, no_speech_prob=0.01, compression_ratio=1.1)


@pytest.fixture
def fake_asr(monkeypatch):
    from app.pipeline import asr

    engines = {"primary": FakeASR("fake:v1", "पहला वाक्य 5"), "secondary": FakeASR("fake2:v1", "पहला वाक्य पाँच")}
    monkeypatch.setattr(asr, "get_primary_asr", lambda settings: engines["primary"])
    monkeypatch.setattr(asr, "get_secondary_asr", lambda settings, language: engines["secondary"])
    return engines


_seeds = itertools.count(100)


@pytest.fixture
def hindi_source(client, tmp_path, fake_asr):
    wav = tmp_path / "phase2.wav"
    seed = next(_seeds)
    make_recording(wav, sentence_layout(seed, n=6), seed=seed)  # unique audio: identical uploads are refused
    sid = upload(client, wav, speaker="p2speaker", gender="male", language="hi").json()["id"]
    run_all_jobs()
    return sid


def test_dual_asr_agreement_and_default_transcript(client, hindi_source):
    clips = [c for c in all_clips(client, hindi_source) if c["status"] == "pending"]
    assert clips
    for c in clips:
        assert c["asr_primary_text"] == "पहला वाक्य 5" and c["asr_secondary_text"] == "पहला वाक्य पाँच"
        assert c["text"] == "पहला वाक्य पाँच"           # normalized primary (numbers spelled out)
        assert c["asr_agreement"] == 0.0                 # digits vs words do not count as disagreement
        assert c["asr_primary_engine"] == "fake:v1" and c["asr_secondary_engine"] == "fake2:v1"
        assert "numbers_expanded" in c["flags"] and "asr_disagreement" not in c["flags"]
        assert c["quality_score"] is not None and c["quality_info"]["method"] == "snr"


def test_retranscribe_preserves_reviewed_and_edited(client, hindi_source, fake_asr):
    clips = [c for c in all_clips(client, hindi_source) if c["status"] == "pending"]
    approved, edited, rejected, untouched = clips[0], clips[1], clips[2], clips[3]
    client.patch(f"/api/clips/{approved['id']}", json={"status": "approved"})
    client.patch(f"/api/clips/{edited['id']}", json={"text": "मैंने खुद लिखा"})
    client.patch(f"/api/clips/{rejected['id']}", json={"status": "rejected", "reject_reason": "noise"})

    fake_asr["primary"] = FakeASR("fake:v2", "दूसरा वाक्य")
    fake_asr["secondary"] = FakeASR("fake2:v2", "बिलकुल अलग बात")
    r = client.post(f"/api/sources/{hindi_source}/retranscribe")
    assert r.status_code == 200
    assert client.post(f"/api/sources/{hindi_source}/retranscribe").status_code == 409  # already queued
    run_all_jobs()

    by_id = {c["id"]: c for c in all_clips(client, hindi_source)}
    a, e, rj, u = (by_id[x["id"]] for x in (approved, edited, rejected, untouched))
    for c in (a, e, rj, u):  # every clip got the new ASR outputs and agreement
        assert c["asr_primary_text"] == "दूसरा वाक्य" and c["asr_primary_engine"] == "fake:v2"
        assert c["asr_agreement"] == 1.5 and "asr_disagreement" in c["flags"]  # 3 edits / 2 reference words
    assert a["status"] == "approved" and a["text"] == "पहला वाक्य पाँच" and not a["needs_review"]
    assert e["status"] == "pending" and e["text"] == "मैंने खुद लिखा" and e["text_edited"]
    assert rj["status"] == "rejected" and rj["reject_reason"] == "noise" and rj["text"] == "पहला वाक्य पाँच"
    assert u["status"] == "pending" and u["text"] == "दूसरा वाक्य" and u["needs_review"]
    assert "numbers_expanded" not in u["flags"]  # text flags were recomputed for the new transcript


def test_interrupted_retranscribe_resumes(client, hindi_source, fake_asr):
    n_clips = len(all_clips(client, hindi_source))
    fake_asr["primary"] = FakeASR("fake:v3", "तीसरा वाक्य", fail_after=2)
    client.post(f"/api/sources/{hindi_source}/retranscribe")
    run_all_jobs()
    src = client.get(f"/api/sources/{hindi_source}").json()
    assert src["status"] == "failed" and "asr" not in src["steps_done"]

    fake_asr["primary"] = FakeASR("fake:v3", "तीसरा वाक्य")
    assert client.post(f"/api/jobs/{src['job']['id']}/retry").status_code == 200
    run_all_jobs()
    clips = all_clips(client, hindi_source)
    assert len(clips) == n_clips and all(c["asr_primary_text"] == "तीसरा वाक्य" for c in clips)
    assert fake_asr["primary"].calls == n_clips - 2   # the two finished clips were not redone
    assert client.get(f"/api/sources/{hindi_source}").json()["status"] == "done"


def test_retranscribe_all(client, hindi_source):
    res = client.post("/api/sources/retranscribe-all").json()
    assert hindi_source in res["queued"]
    run_all_jobs()


def test_export_script_both_and_hf_dataset(client, portal_env, hindi_source):
    from datasets import Audio, load_from_disk

    ids = [c["id"] for c in all_clips(client, hindi_source) if c["status"] == "pending"][:3]
    client.post("/api/clips/bulk", json={"ids": ids, "action": "approve"})
    r = client.post("/api/exports", json={"name": "both_scripts", "formats": ["jsonl", "csv", "hf"],
                                          "script": "both", "languages": ["hi"]})
    export_id = r.json()["id"]
    run_all_jobs()
    exp = client.get(f"/api/exports/{export_id}").json()
    assert exp["status"] == "done", exp["error"]
    out = portal_env / "data" / exp["out_dir"]
    rows = [json.loads(x) for x in (out / "train.jsonl").read_text(encoding="utf-8").splitlines()]
    manifest = [json.loads(x) for x in (out / "manifest.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 2 * exp["clip_count"]
    assert {m["script"] for m in manifest} == {"devanagari", "roman"}
    roman = [r["text"] for r, m in zip(rows, manifest) if m["script"] == "roman"]
    assert all(t.startswith("p2speaker: ") and not any("ऀ" <= ch <= "ॿ" for ch in t) for t in roman)
    assert len({r["audio"] for r in rows}) == exp["clip_count"]  # both rows share one wav

    ds = load_from_disk(str(out / "hf_dataset"))
    assert ds.num_rows == len(rows)
    assert set(ds.column_names) >= {"audio", "text", "transcript", "speaker", "language", "gender", "script"}
    first = ds.cast_column("audio", Audio(decode=False))[0]
    x, sr = sf.read(io.BytesIO(first["audio"]["bytes"]))
    assert sr == 24000 and x.ndim == 1 and first["text"].startswith("p2speaker: ")
    stats = json.loads((out / "stats.json").read_text())
    assert stats["rows_per_script"] == {"devanagari": exp["clip_count"], "roman": exp["clip_count"]}


def test_original_audio_variant_requires_denoised_source(client, hindi_source):
    clip_id = all_clips(client, hindi_source)[0]["id"]
    assert client.get(f"/api/clips/{clip_id}/audio?variant=original").status_code == 404
    assert client.get(f"/api/clips/{clip_id}/audio").status_code == 200


def test_low_quality_clips_are_auto_rejected(client, tmp_path, fake_asr):
    from app.config import get_settings

    q = get_settings().quality
    saved = q.reject_below, q.review_below
    q.reject_below, q.review_below = 5.5, 6.0  # above the 1-5 scale: everything is "low quality"
    try:
        wav = tmp_path / "noisy.wav"
        make_recording(wav, sentence_layout(11, n=4), seed=11)
        sid = upload(client, wav, speaker="p2speaker", gender="male", language="hi").json()["id"]
        run_all_jobs()
    finally:
        q.reject_below, q.review_below = saved
    clips = all_clips(client, sid)
    assert clips and all(c["status"] == "rejected" for c in clips)
    assert {c["reject_reason"] for c in clips} <= {"low_quality", "too_short"}
    assert any(c["reject_reason"] == "low_quality" for c in clips)
