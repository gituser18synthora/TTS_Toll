# Orpheus TTS Dataset Portal

A local web portal that turns long raw recordings into a clean, human-reviewed,
training-ready dataset for fine-tuning **Orpheus TTS** (Canopy Labs, Llama-3B + SNAC 24 kHz),
for multi-speaker Indian TTS in Hindi, English, Hinglish, Tamil, Telugu and Malayalam.

```
upload ──► convert 24 kHz mono ──► DeepFilterNet denoise ──► Silero VAD ──► [multi-speaker file: diarize → you name the voices]
   ──► cut at pauses (2–15 s, one speaker per clip) ──► trim · −23 LUFS · −1 dBFS ──► DNSMOS quality (+ music/babble tagger)
   ──► ASR #1 (whisper) ──► ASR #2 (IndicConformer / whisper) + WER agreement ──► normalize text
   ──► speaker-consistency check + duplicate detection ──► review UI ──► export ──► pretrain-check
```

| Phase | Scope | Status |
|---|---|---|
| 1 | upload + metadata, convert, VAD segmentation, loudness normalization, DB, single ASR, review UI with shortcuts, JSONL export | **done** |
| 2 | DeepFilterNet denoise, DNSMOS quality + auto-reject, music/babble flags, dual ASR + agreement, numbers→words, script handling + romanized export, re-transcribe job, dashboard, HF `datasets` export | **done** |
| 3 | diarization + voice naming for multi-speaker files, speaker-consistency check, duplicate detection, Docker (CPU / GPU), `pretrain-check`, `dump-numbers`, final docs | **done** |

---

## 1. Local setup (CPU or small GPU)

Requirements: **Python 3.11**, **ffmpeg**, **Node.js ≥ 20.19** (only for building the UI), ~5 GB disk for models.
Prefer containers? See [§7 Docker](#7-docker).

```bash
cp .env.example .env  # then put your HuggingFace token in it (optional, see below)
./setup.sh            # venv (.venv) + pinned deps + UI build + DB init + model download
./run_dev.sh          # API :8000 + worker + UI :5173  (Ctrl+C stops all three)
```

Open **http://localhost:5173**. On WSL2 the Windows browser reaches it through localhost forwarding.

* `setup.sh` auto-detects an NVIDIA GPU and installs the matching torch wheels. Force it with `--cpu` / `--gpu`.
* Without `python3.11` on PATH it uses `uv` if installed, or `PYTHON=/path/to/python3.11 ./setup.sh`.
* Why Python 3.11: DeepFilterNet's compiled library only has wheels up to 3.11, and NeMo / pyannote are
  best-tested there.
* `./run_dev.sh --prod` builds the UI once and serves UI + API from port 8000 (no Vite). This is handy on a server.

### HuggingFace token (`.env`) and the two gated models

Everything works without a token; two models are better with one. Both are **gated**: with your HuggingFace
account open the model page and click **Agree and access repository**, then put that account's token in `.env`
as `HF_TOKEN=hf_…`. The file is git-ignored and loaded with python-dotenv; its values override the shell
environment. The token is never logged or copied anywhere.

| Gated model | Used for | Without access the portal uses |
|---|---|---|
| [ai4bharat/indic-conformer-600m-multilingual](https://huggingface.co/ai4bharat/indic-conformer-600m-multilingual) | second ASR for hi / ta / te / ml | a second whisper pass (`medium`) |
| [pyannote/speaker-diarization-community-1](https://huggingface.co/pyannote/speaker-diarization-community-1) | diarization of multi-speaker files, overlap detection | clustering of speaker embeddings (no overlap detection) |

Check access: `cd backend && ../.venv/bin/python -m app.cli check-asr --lang hi` (IndicConformer) and
`python -m app.cli download-models`, which reports "no access" per model. The job log and the Dashboard
("Models in use") always say which engine ran and why.

**Privacy:** pyannote.audio 4 sends usage telemetry to pyannote's servers by default. The portal turns it off
(`PYANNOTE_METRICS_ENABLED=false`) along with HuggingFace Hub telemetry (`HF_HUB_DISABLE_TELEMETRY=1`); audio never
leaves the machine.

Run pieces individually (from `backend/`, with `../.venv/bin/python`):

```bash
python -m uvicorn app.main:app --port 8000      # API (+ built UI if frontend/dist exists)
python -m app.worker                            # job worker (required for processing!)
python -m app.cli doctor                        # what is installed / which device is used
python -m app.cli download-models [--all]       # pre-fetch every model the config uses into ./models
python -m app.cli check-asr --lang hi           # load the secondary ASR and transcribe one clip
python -m app.cli dump-numbers --lang ta        # how numbers are spelled, for native-speaker review (§3)
python -m app.cli pretrain-check <export>       # validate an export before training (§6)
python -m pytest                                # 132 tests, ~30 s; model-backed ones skip if weights are missing
```

The header shows **worker online / offline**; hover it to see which models the worker is using.
Uploads are accepted without a worker, but nothing gets processed until one runs.

## 2. Using the portal

1. **Dashboard**: approved hours (the number you train on), the pending backlog, flagged clips and mean quality.
   Charts show hours per speaker and per language by review state, the DNSMOS distribution with the
   thresholds marked, review progress per day, why clips are flagged and why they were rejected, the models
   in use and running jobs. Every chart has a table view.
2. **Upload**: drop one or more files. Each needs speaker name, gender and language (notes optional).
   Rule: one upload = one speaker, unless you tick **Multiple speakers in this file** (then only the language
   is needed, see below). The speaker name becomes the training prefix (`ravi: …`), so it must not contain `:`
   or `|`. Existing speakers are reused; a gender mismatch is refused. Identical files (same SHA-256) are
   refused. Originals are stored untouched under `data/raw/<speaker>/` (`data/raw/_multi_speaker/`).
3. **Files & Jobs**: live progress per file, job log, **Retry** (resumes from the last completed step),
   **Reprocess** (deletes this file's clips and derived audio and regenerates them with the current config;
   the original is kept), **Re-transcribe** (see below), **Cancel**, and **Edit** metadata (speaker/language
   changes propagate to clips). Multi-speaker files show **Assign speakers** / **Speakers**. Two buttons work
   across the whole dataset: **Re-check speakers** (voice-consistency check for every speaker) and
   **Find duplicates**.
4. **Review**: filter by status / flagged / speaker / language / file / flag / text, and sort by ASR
   confidence, duration, quality or ASR WER. Check rows for bulk approve / reject / reset. The transcript
   auto-saves, and the approved text is exactly what gets exported.
   * Both ASR outputs are shown with word-level differences. **use primary** / **use secondary** copies one
     into the transcript box.
   * For denoised files, an **original / denoised** toggle (key <kbd>O</kbd>) lets you hear the clip before
     enhancement, level-matched, to catch denoising damage.
   * **voice match** is the clip's similarity to the speaker's other clips (§3). A duplicate links to the clip
     that was kept instead.
   * Hover a flag to see what it means.

   | Key | Action |
   |---|---|
   | <kbd>Space</kbd> | play / pause |
   | <kbd>A</kbd> | approve (then next clip; toggle "next clip after approve/reject") |
   | <kbd>R</kbd> | reject with the reason selected next to the button |
   | <kbd>Enter</kbd> | save transcript & next (also inside the transcript box) |
   | <kbd>←</kbd> / <kbd>→</kbd> | previous / next clip (pages load automatically) |
   | <kbd>E</kbd> | edit transcript (cursor at end) |
   | <kbd>Ctrl</kbd>+<kbd>Enter</kbd> | approve & next, even while typing |
   | <kbd>Esc</kbd> | leave the transcript box |
   | <kbd>P</kbd> | replay from start |
   | <kbd>O</kbd> | original / denoised audio |

   Shortcuts never fire while you type in a text field. They also wait for Indic IME composition
   (Hindi/Tamil/Telugu/Malayalam keyboards) to finish, so Enter only commits the IME.
   Fast loop: listen → <kbd>A</kbd>. If the text is wrong: <kbd>E</kbd> → fix → <kbd>Ctrl</kbd>+<kbd>Enter</kbd>.
5. **Export**: pick speakers / languages (none selected = all), the Hindi/Hinglish script
   (**Devanagari / Roman / Both**), and the formats (JSONL always; LJSpeech CSV and HuggingFace dataset
   optional). See the hours preview, then export. Only **approved** clips with a non-empty transcript are
   exported, each audio only once.

### Multi-speaker recordings (interviews, dialogues, podcasts)

1. Upload with **Multiple speakers in this file** ticked (language only).
2. The worker converts, denoises, runs VAD and **diarizes**, then pauses the file in **needs speakers**.
3. Press **Assign speakers**. Each detected voice (`SPEAKER_00` talks most) shows its minutes and up to three
   short samples. Give each voice a name and gender (existing speakers are reused, so one person's clips from
   many files end up under one name), or tick **ignore** (an interviewer, a caller you don't want). Giving two
   labels the same name merges them: use it when the diarizer split one person in two.
4. **Save speakers**: processing continues and the clips are cut, one speaker per clip.

Rules that keep the clips clean:

* A clip never contains two voices. Cuts are forced at every speaker change, and padding stops halfway into the
  pause before another voice.
* Speech of ignored voices, and speech that could not be attributed with confidence (a lone syllable between
  pauses, often the other speaker's "haan"), is never in any clip, and no clip is merged across it. The job log
  says how many seconds were left out.
* With pyannote, overlapped speech is detected: a clip with more than 0.1 s of it is flagged `overlap`, and
  rejected when overlap covers over 30 % of it.
* Once clips exist, **Speakers** lets you rename voices; the speaker check then re-runs. Changing what is
  ignored needs **Reprocess** (the file is re-diarized and you name the voices again).

Without pyannote access, the clustering fallback is used. It clusters wespeaker embeddings of 1.5 s windows
(cosine threshold 0.7) and folds tiny clusters into the nearest main voice. On a 78 s test dialogue
(Hindi male / English female narrator, 8 turns) it found both voices and attributed 100 % of the speech
correctly, and no clip contained the other voice. It suits dialogues with distinct voices; for crosstalk-heavy
recordings accept the pyannote terms.

### Re-transcribe (after moving to the GPU server)

**Re-transcribe** (per file, or **Re-transcribe all**) re-runs both ASR systems and the agreement score on the
existing clips **without re-segmenting**:

* every clip gets the new ASR outputs, engine names, WER and ASR flags;
* **approved and rejected clips keep their status and their transcript**;
* **transcripts you edited are never overwritten**;
* only *pending, never-edited* transcripts take the new primary text.

It is resumable: if it stops, Retry continues with the clips that are not done yet.
Typical use: process and review locally with `small` on CPU, `rsync` to the GPU box, then press
**Re-transcribe all** to get `large-v3` + IndicConformer output on everything still pending.

## 3. What the pipeline does (and the defaults it uses)

| Step | Details |
|---|---|
| convert | ffmpeg → 24 kHz mono 16-bit WAV (`soxr` resampler when available) → `data/converted/src_XXXXX.wav` (kept for A/B listening) |
| denoise | **DeepFilterNet3** on the original decoded at 48 kHz, in 30 s chunks with 1 s context and cross-faded seams (memory stays flat for any length), then → 24 kHz `data/enhanced/src_XXXXX.wav`. About 30× real-time on CPU. Skipped with a warning if unavailable. `denoise.atten_lim_db` keeps a little natural noise floor if full suppression sounds processed. |
| VAD | Silero VAD v6 (batched ONNX, about 500× real-time on CPU) on the enhanced audio → `data/work/…/vad.npz`. Energy VAD fallback. |
| diarize | multi-speaker files only: pyannote community-1 or the clustering fallback (§2) → `work/…/diarization.json` (turns, overlaps) + `speaker_samples/`; then the file waits for you to name the voices |
| segment | Speech regions are split at pauses ≥ `min_silence_ms`. A dynamic program then picks cut points so that clips are 2–15 s (hard max 20 s), cuts land on the **longest available pauses**, speech is never merged across a pause > `max_pause_in_clip`, and lengths lean toward `target_duration`. Clips only start/end inside silence. Speech running longer than 15 s with no real pause is split at the deepest dip in speech probability and flagged `forced_cut`. 150 ms padding each side. Multi-speaker files: never across a speaker change or excluded speech. |
| per clip | trim silence beyond the padding · loudness → −23 LUFS (BS.1770) · look-ahead limiter guaranteeing ≤ −1 dBFS · 16-bit WAV → `data/clips/src_XXXXX/` |
| quality | **DNSMOS P.835** (SIG / BAK / OVRL; `quality_score` = OVRL, 1–5). OVRL < 2.0 → `rejected / low_quality`; < 2.9 → flag `borderline_quality` (tune both after looking at the Dashboard's score distribution for your recordings). SNR-estimate fallback on the same scale. An AudioSet tagger (AST) flags `music` and `babble` (background talk / overlap). It runs on GPU by default and is skipped on CPU, where it takes ~2 s per clip. |
| ASR #1 | faster-whisper with the language hint (`hinglish` → `hi`): `small` int8 on CPU, `large-v3` fp16 on GPU. Flags: `low_asr_confidence`, `asr_no_speech`, `asr_repetition` (hallucination), `sparse_transcript`, `empty_transcript`. |
| ASR #2 | **IndicConformer** (hi/ta/te/ml, CTC decoding) or a second whisper pass with a different size (`medium`) for en/hinglish and as the fallback. `asr_agreement` = WER between the two after normalization (case, punctuation, digits vs words, nukta, candrabindu and हिन्दी/हिंदी spelling ignored). Above the per-language threshold → flag `asr_disagreement`. |
| text | Unicode/punctuation cleanup, `[music]`-style tags removed (`annotation_removed`), numbers → words in the clip's language (`numbers_expanded`; lakh/crore grouping, years, currency, %, decimals, phone numbers digit by digit), characters outside the language's script removed (`chars_removed`, `wrong_script` when e.g. Hindi came back in Urdu script), danda `।` as the Hindi/Hinglish sentence end. |
| speaker check | Every clip gets a 256-d voice embedding (`pyannote/wespeaker-voxceleb-resnet34-LM`, not gated). Per speaker, a robust centroid over all their clips from all files gives each clip's **voice match** (cosine). Below `speaker_check.min_similarity` (0.5), or, from 20 clips on, an outlier below median − 4·MAD → flag `speaker_mismatch` (pending clips go to review). It catches a wrongly named upload, a second voice, or a wrong diarization label. Runs once a speaker has ≥ 5 clips; **Re-check speakers** re-runs it everywhere. |
| dedup | **exact**: identical audio samples. **near**: same speaker, same normalized transcript (≥ 3 words), duration within 10 %, and the waveforms are the same recording (cross-correlation ≥ 0.8). A re-encoded or re-trimmed copy scores 0.9–1.0, while a second *take* of the sentence scores far lower, so genuine repeated readings are kept. The best-reviewed clip is kept (approved > pending > rejected, then oldest); pending duplicates become `rejected / duplicate` with a link to it. Runs per file against the whole dataset; **Find duplicates** re-runs everything. |
| bounds | clips < `min_duration` → `rejected / too_short`, > `hard_max_duration` → `rejected / too_long`. Audio is **never deleted**, only marked. |

Every step is checkpointed (`source_files.steps_done`). A crash, Ctrl+C or failure resumes at the
first unfinished step, and ASR resumes per clip. Jobs are stored in the database. A worker that died is
detected by heartbeat/PID and its job re-queued.

### Script policy (Hindi / Hinglish)

* **Stored text is canonical Devanagari.** English words stay in Latin script exactly as the ASR wrote them
  (`Dr. Sharma ने meeting cancel कर दी।`). Numbers are spelled out in Hindi for both hi and hinglish
  (`text.numbers_language`).
* **Romanization happens only at export** (`script: roman` or `both`). It uses indic-transliteration plus Hindi
  schwa deletion (कमल → kamal, समझना → samajhna, बचपन → bachpan) and the casual spelling people type
  (`text.roman_scheme: hinglish` → *aaj mausam bahut achha hai*). Alternatives are `iso`, `iast` and `itrans`.
  Latin words pass through unchanged.
* `both` writes one row per script for each Hindi/Hinglish clip, both pointing at the same wav.

### Checking how numbers are spelled (`dump-numbers`)

```bash
cd backend && ../.venv/bin/python -m app.cli dump-numbers --lang hi    # hi | hinglish | ta | te | ml | en
# -> data/reports/numbers_hi.tsv   (--out file.tsv, --scheme iso for another roman spelling)
```

It writes ~45 representative numbers exactly as the pipeline spells them: 0–20, tens, 100, 1000, 1 lakh,
1 crore, decimals (2.5, 1.87), years (2024), currency (₹500), plus compound forms. Hindi/Hinglish rows also show
the romanized export spelling. The TSV opens in Excel / Google Sheets with empty *correct? (y/n)* and
*correction* columns for a native speaker:

```
#   category  written  spoken (as the portal writes it)  roman export (hinglish)  note  correct? (y/n)  correction
20  0-20      19       उन्नीस                             unnees
29  hundred   100      एक सौ                              ek sau
```

## 4. Configuration (`config.yaml`)

| Key | Default | Meaning |
|---|---|---|
| `paths.data_root` | `./data` | DATA_ROOT: everything the portal stores (also `PORTAL_DATA_ROOT`) |
| `paths.models_dir` | `./models` | whisper / HF / DNSMOS / DeepFilterNet weights (also `PORTAL_MODELS_DIR`) |
| `device` | `auto` | `auto` \| `cpu` \| `cuda` (also `PORTAL_DEVICE`) |
| `worker.embedded` | `false` | run the worker inside the API process instead of separately |
| `vad.threshold` / `min_silence_ms` | 0.5 / 150 | speech threshold; shortest pause usable as a cut |
| `segmentation.min/max/hard_max_duration` | 2 / 15 / 20 s | clip length bounds |
| `segmentation.target_duration` | 8 s | weak preference among equally natural cut plans |
| `segmentation.padding_ms` | 150 | silence kept around speech |
| `segmentation.max_pause_in_clip` | 1.2 s | never merge speech across longer pauses |
| `denoise.engine` / `atten_lim_db` | deepfilternet / null | `none` disables; e.g. `25` keeps a faint noise floor |
| `diarization.engine` / `model` | pyannote / speaker-diarization-community-1 | `clustering` always uses the fallback |
| `diarization.max_speakers` | 8 | upper bound on detected voices |
| `diarization.cluster_threshold` | 0.7 | fallback only: cosine distance within which windows are one voice (lower → more voices) |
| `diarization.overlap_flag_sec` / `overlap_reject_ratio` | 0.1 s / 0.3 | pyannote only: flag `overlap` / reject above this share of the clip |
| `clip.target_lufs` / `peak_dbfs` | −23 / −1 | loudness target and limiter ceiling |
| `quality.reject_below` / `review_below` | 2.0 / 2.9 | DNSMOS OVRL auto-reject / flag thresholds |
| `quality.tagger.enabled` | auto | music/babble tagger: auto = GPU only, or `true` / `false` |
| `speaker_check.enabled` / `min_similarity` / `min_clips` | true / 0.5 / 5 | voice-consistency check (same voice scores ~0.75–0.9, other voices ~0.1) |
| `dedup.enabled` / `min_similarity` | true / 0.8 | duplicate detection; waveform correlation for "same recording" |
| `asr.primary.model_cpu` / `model_gpu` | small / large-v3 | primary whisper size |
| `asr.secondary.engine` | indicconformer | `indicconformer` \| `faster-whisper` \| `none` |
| `asr.secondary.indicconformer.languages` | hi, ta, te, ml | languages sent to IndicConformer |
| `asr.secondary.whisper_cpu` / `whisper_gpu` | medium / medium | second whisper pass (en, hinglish, fallback) |
| `asr.agreement.max_wer` | per language | WER above which `asr_disagreement` is flagged |
| `asr.transcript_source` | primary | per language: which output pre-fills the transcript (`{ta: secondary}` …) |
| `asr.review.*` | | thresholds that flag clips `needs_review` |
| `languages.<code>.whisper` | | ASR language hint per portal language; optional `initial_prompt` |
| `text.roman_scheme` | hinglish | romanization used by `script: roman / both` exports |
| `text.numbers_language` | {hinglish: hi} | whose words digits are read in |

Restart `run_dev.sh` after editing (the worker reads the config once). Use **Reprocess** to apply new
segmentation/denoise/diarization settings to files that were already processed, **Re-transcribe** for new ASR
settings, and **Re-check speakers** / **Find duplicates** for the speaker-check and dedup thresholds.

### Speed (measured on this 8-core laptop CPU; the GPU server is much faster)

| Stage | CPU | Notes |
|---|---|---|
| denoise | ~30× real-time | DeepFilterNet3 |
| VAD + cutting | ~100× real-time | |
| diarization (clustering fallback) | ~9× real-time | 78 s dialogue in 9 s, model load included |
| DNSMOS | ~0.1 s per clip | |
| speaker embeddings | ~0.3 s per clip | |
| whisper `small` (ASR #1) | 1.3–2.5× real-time | Hindi slower than English |
| whisper `medium` (ASR #2 fallback) | 0.3× real-time | the slowest step on CPU. Set `asr.secondary.engine: none` for quick local runs, or get IndicConformer access. |

`base` is not offered as a fallback: in testing it wrote Hindi audio in **Urdu script** even with the `hi` hint.

## 5. Data folder layout and moving to the GPU server

```
data/                         ← DATA_ROOT (all DB paths are relative to this folder)
├── portal.db                 SQLite (WAL): speakers, source files, clips, jobs, exports
├── raw/<speaker>/…           original uploads, never modified (multi-speaker files: raw/_multi_speaker/)
├── converted/src_XXXXX.wav   24 kHz mono PCM16 (un-denoised, for A/B listening)
├── enhanced/src_XXXXX.wav    DeepFilterNet output the clips are cut from
├── work/src_XXXXX/           vad.npz (re-segmentation without re-running VAD);
│                             multi-speaker: diarization.json + speaker_samples/
├── clips/src_XXXXX/*.wav     final clips
├── exports/<name>/ + <name>.zip
├── reports/numbers_<lang>.tsv  dump-numbers output
└── logs/api.log, worker.log
models/                       ← model weights (optional to copy; `app.cli download-models` refetches)
.env                          ← your HF token (copy it yourself; never committed)
```

Because every stored path is relative, the whole state moves with one folder:

```bash
# on the dev machine: stop run_dev.sh first so the SQLite WAL is checkpointed
rsync -a --info=progress2 data models user@gpu-server:/srv/orpheus-portal/   # no trailing slashes
# on the server (project copied or cloned to /srv/orpheus-portal)
cd /srv/orpheus-portal && ./setup.sh --gpu && ./run_dev.sh --prod     # or Docker, §7
# then: Files & Jobs → "Re-transcribe all" to upgrade pending transcripts to large-v3 + IndicConformer
```

On the server `device: auto` picks CUDA (ASR becomes `large-v3`, the music/babble tagger turns on).
Reach the UI through an SSH tunnel (`ssh -L 8000:127.0.0.1:8000 user@gpu-server`) rather than exposing the
port. The portal has no authentication. IndicConformer runs on ONNX Runtime; for GPU inference install
`onnxruntime-gpu` in place of `onnxruntime` on the server (CPU works, just slower).

To only train on the server, the export zip is enough: `scp data/exports/<name>.zip server:` then
`unzip <name>.zip && cd <name> && sha256sum -c SHA256SUMS`.

## 6. Training data format (Orpheus / Unsloth)

Each export folder (and zip) contains:

```
<name>/
├── train.jsonl     {"audio": "wavs/<id>.wav", "text": "<speaker_name>: <final transcript>"}
├── manifest.jsonl  + speaker, gender, language, script, duration, clip/source ids, scores, flags
├── metadata.csv    LJSpeech-style: id|text|normalized_text|speaker|language|script
├── hf_dataset/     (optional) datasets.save_to_disk, audio bytes embedded
├── wavs/           24 kHz mono 16-bit PCM
├── stats.json      totals per speaker / language / script
├── export.log      what was included / skipped / fixed
├── README.txt
└── SHA256SUMS
```

Example `train.jsonl` lines (`script: both`):

```json
{"audio": "wavs/ravi_src_00012_00034.wav", "text": "ravi: आज मौसम बहुत अच्छा है।"}
{"audio": "wavs/ravi_src_00012_00034.wav", "text": "ravi: aaj mausam bahut achha hai."}
{"audio": "wavs/meera_src_00031_00002.wav", "text": "meera: Let's meet at the station tomorrow."}
```

The `"<speaker>: text"` prefix is the Orpheus multi-speaker prompt format. At inference you prompt with the same
`"ravi: …"`. Load it for the Unsloth Orpheus notebook (or Canopy's finetune scripts) like this:

```python
import os
from datasets import load_dataset, load_from_disk, Audio

root = "/data/orpheus_20261006"                     # unzipped export folder
ds = load_dataset("json", data_files=f"{root}/train.jsonl", split="train")
ds = ds.map(lambda r: {"audio": os.path.join(root, r["audio"])})
ds = ds.cast_column("audio", Audio(sampling_rate=24000))
# or, if exported with the HuggingFace option (audio embedded, no paths):
ds = load_from_disk(f"{root}/hf_dataset")           # columns: audio, text, transcript, speaker, gender, language, script, ...
# ds[0] -> {"audio": {"array": ..., "sampling_rate": 24000}, "text": "ravi: …"}
```

In the Unsloth notebook the text is **already prefixed**. Do not add a `source` column, or the speaker name is
prepended twice. Decoding audio with `datasets` ≥ 4 needs `torchcodec` on the training machine. The SNAC
tokenisation step (`tokenise_audio(example["audio"]["array"])`) works unchanged because the audio is already
24 kHz mono.

### Before training: `pretrain-check`

```bash
cd backend && ../.venv/bin/python -m app.cli pretrain-check orpheus_20261006   # name, folder or .zip
```

It needs only the export (no database), so it also runs on the training server. **Errors** (exit code 1):

* `train.jsonl` that doesn't parse, or rows with keys other than `audio` / `text`;
* text that is empty, multi-line, or missing the `speaker: ` prefix (checked against the manifest);
* a missing or unreadable wav, or one that isn't 24 kHz mono 16-bit PCM;
* a duration outside `segmentation.min_duration` … `hard_max_duration`;
* a clip flagged `wrong_script` or `needs_review`, or text mostly in another script than its language's
  (e.g. Hindi written in Urdu letters);
* duplicate rows (same text + audio), identical audio in two wav files, or a SHA256SUMS mismatch.

**Warnings**: a speaker with under 30 min of audio (`--min-speaker-minutes`), stray letters from another script,
and a missing manifest or checksums. The summary shows minutes, clips and rows per speaker and per language:

```
speaker                   minutes   clips   rows   per language (min)
ravi                         41.2     402    804   hi 30.3, hinglish 10.9
meera                        12.5     118    118   en 12.5
TOTAL                        53.7     520    922

WARN  low_speaker_data (1)
        speaker 'meera' has only 12.5 min (< 30 min) of approved audio

RESULT: OK  (0 errors, 1 warnings)
```

Options: `--min-duration` / `--max-duration`, `--no-checksums`, `--json report.json`, `--max-examples N`.

## 7. Docker

The image holds the API, the built UI and the worker; `docker compose` runs them as two containers.
Data, models, `config.yaml` and `.env` stay on the host, so the folder still moves as one unit.

```bash
docker compose up -d --build                  # CPU: UI + API on http://127.0.0.1:8000, worker alongside
docker compose logs -f worker
docker compose exec worker python -m app.cli download-models
docker compose exec worker python -m app.cli pretrain-check <export-name>
docker compose down                           # the running job is re-queued and resumes on the next start

# GPU server (NVIDIA driver + NVIDIA Container Toolkit installed)
TORCH_VARIANT=cu128 docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d --build
```

| Variable | Default | Meaning |
|---|---|---|
| `PORTAL_DATA_DIR` | `./data` | host folder mounted as DATA_ROOT (`/data`) |
| `PORTAL_MODELS_HOST_DIR` | `./models` | host folder for model weights (`/models`) |
| `PORTAL_PORT` | 8000 | published on 127.0.0.1 only (the portal has no login; use an SSH tunnel) |
| `PORTAL_UID` / `PORTAL_GID` | 1000 | user the containers run as, so files in `data/` stay yours (`id -u`, `id -g`) |
| `TORCH_VARIANT` | `cpu` | `cpu`, `cu126` or `cu128` torch wheels (CUDA 12 is required by ctranslate2) |
| `APT_CLOCK_SKEW_OK` | `false` | build-time workaround for a wrong system clock (see §8) |

The image is python:3.11-slim plus ffmpeg (3.8 GB for the CPU variant). `.env` is read at runtime and never
copied into the image. Verified here: a 20 s Hindi upload was processed by the worker container (201 s on CPU),
exported, and passed `pretrain-check` inside the container; `docker compose down` takes ~7 s.

## 8. Troubleshooting

* **Worker offline**: start `run_dev.sh`, or `cd backend && ../.venv/bin/python -m app.worker`.
* **Job failed**: open *Log* on the Files & Jobs page. Fix the cause, then press *Retry*; it resumes.
* **A file stays in "needs speakers"**: that is the pause for naming voices. Press **Assign speakers**.
* **"… no access …" for IndicConformer or diarization**: the token works, but that account hasn't accepted the
  model terms (see §1). "HF_TOKEN not set" means `.env` is missing the token.
* **The clustering fallback finds too many voices**: give the extra labels the name of the person they belong to
  (that merges them), or raise `diarization.cluster_threshold` and **Reprocess**. Too few voices: lower it.
* **Many `speaker_mismatch` flags for one speaker**: listen to a few. If they really are that person (very
  different mic or room), lower `speaker_check.min_similarity` and press **Re-check speakers**. Otherwise a file
  was uploaded under the wrong name: fix it with **Edit**.
* **`pip check` complains about deepfilternet / numpy<2**: expected. DeepFilterNet pins outdated versions and is
  installed with `--no-deps` from `requirements-nodeps.txt`; `tests/test_models_optional.py` verifies it works.
* **GPU: `libcublas.so.12` / cuDNN not found**: ctranslate2 needs CUDA 12 libraries. Use the cu126/cu128 torch
  wheels (`setup.sh` and the Docker build do), and start through `run_dev.sh` (or the container entrypoint),
  which puts the pip-installed NVIDIA libraries on `LD_LIBRARY_PATH`.
* **Docker build: "Release file … is not valid yet"**: the system clock is behind (common on WSL2 after sleep).
  Fix the clock with `sudo hwclock -s`, or `wsl --shutdown` from Windows. For a one-off build:
  `APT_CLOCK_SKEW_OK=true docker compose build`, which skips apt's date check during that build only.
* **Docker: "port is already allocated"**: another program uses the port; start with `PORTAL_PORT=8765`.
* **Keep `data/` on a Linux filesystem**. On WSL that means not under `/mnt/c`, because SQLite locking over 9p is unreliable and slow.
* **"database busy"**: just repeat the action. Writers wait up to 30 s for each other, so this only shows up under extreme load.

## 9. Project structure

```
backend/app/
  main.py            FastAPI app (API under /api, serves frontend/dist at /)
  config.py          typed config.yaml + .env loader      paths.py   DATA_ROOT layout
  db.py / models.py  SQLite engines (WAL, BEGIN IMMEDIATE writers), ORM, in-place migrations
  engines.py         which model each component actually runs with (shown in the UI)
  cli.py             init · doctor · download-models · check-asr · dump-numbers · pretrain-check · worker
  api/               uploads, sources (+ speaker naming, re-transcribe), clips (+ A/B audio), jobs, exports,
                     speakers, maintenance (speaker check, dedup), system
  pipeline/          convert · denoise · vad · diarize · segment · audio_ops · quality · asr · transcripts
                     · speakers · dedup · text_norm · numbers · translit · wer · orchestrator
  worker/            DB-backed queue (claim / heartbeat / cancel / resume) + worker loop
  export/            exporter.py (JSONL / CSV / HF dataset / checksums / zip, script variants) · pretrain_check.py
backend/tests/       132 tests: segmentation, audio ops, numbers & romanization, text/WER, end-to-end, dual ASR,
                     re-transcribe rules, exports, diarization flow, speaker check, dedup, pretrain-check,
                     migrations, real-model checks (DNSMOS, DeepFilterNet)
frontend/src/        React 19 + Vite + Tailwind 4 + wavesurfer.js 8: Dashboard, Upload, Files & Jobs, Review, Export
Dockerfile  docker-compose.yml  docker-compose.gpu.yml  docker/entrypoint.sh  .dockerignore
setup.sh  run_dev.sh  config.yaml  .env.example  requirements*.txt  constraints.txt (full lock)
```
