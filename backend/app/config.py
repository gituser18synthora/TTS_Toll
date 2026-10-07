"""Typed configuration loaded from config.yaml.

Every section has defaults, so a partial config.yaml works. Relative paths are
resolved against the directory that contains the config file.
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class _Section(BaseModel):
    model_config = ConfigDict(extra="forbid")


class PathsCfg(_Section):
    data_root: str = "./data"
    models_dir: str = "./models"


class ServerCfg(_Section):
    host: str = "127.0.0.1"
    port: int = 8000
    cors_origins: list[str] = ["http://localhost:5173", "http://127.0.0.1:5173"]


class WorkerCfg(_Section):
    embedded: bool = False
    poll_interval_sec: float = 2.0
    stale_job_sec: float = 180.0


class UploadCfg(_Section):
    allowed_extensions: list[str] = ["wav", "mp3", "m4a", "flac", "ogg"]
    max_file_gb: float = 4.0


class AudioCfg(_Section):
    sample_rate: int = 24000
    ffmpeg: str = "ffmpeg"
    ffprobe: str = "ffprobe"


class VadCfg(_Section):
    engine: Literal["silero", "energy"] = "silero"
    threshold: float = Field(0.5, gt=0, lt=1)
    neg_threshold: float | None = None
    min_speech_ms: int = Field(250, ge=0)
    min_silence_ms: int = Field(150, ge=32)

    @property
    def exit_threshold(self) -> float:
        if self.neg_threshold is not None:
            return self.neg_threshold
        return max(0.01, self.threshold - 0.15)


class SegmentationCfg(_Section):
    min_duration: float = 2.0
    max_duration: float = 15.0
    hard_max_duration: float = 20.0
    target_duration: float = 8.0
    padding_ms: int = 150
    max_pause_in_clip: float = 1.2

    @model_validator(mode="after")
    def _check_bounds(self) -> "SegmentationCfg":
        if not 0 < self.min_duration < self.max_duration <= self.hard_max_duration:
            raise ValueError("segmentation: need 0 < min_duration < max_duration <= hard_max_duration")
        if not self.min_duration <= self.target_duration <= self.max_duration:
            raise ValueError("segmentation: target_duration must lie within [min_duration, max_duration]")
        if not 0 <= self.padding_ms <= 1000:
            raise ValueError("segmentation: padding_ms must be within 0..1000")
        if self.max_pause_in_clip <= 0:
            raise ValueError("segmentation: max_pause_in_clip must be > 0")
        return self


class DenoiseCfg(_Section):
    engine: Literal["deepfilternet", "none"] = "deepfilternet"
    atten_lim_db: float | None = None  # e.g. 25: keep a little natural noise floor, fewer artifacts
    chunk_sec: float = 30.0


class TaggerCfg(_Section):
    enabled: Literal["auto"] | bool = "auto"  # auto = only when running on a GPU
    model: str = "MIT/ast-finetuned-audioset-10-10-0.4593"
    music_threshold: float = 0.30
    babble_threshold: float = 0.30


class QualityCfg(_Section):
    engine: Literal["dnsmos", "snr"] = "dnsmos"
    reject_below: float = 2.0   # DNSMOS OVRL (1-5): auto-reject as low_quality
    review_below: float = 2.9   # flag for a listen
    tagger: TaggerCfg = TaggerCfg()


class DiarizationCfg(_Section):
    engine: Literal["pyannote", "clustering"] = "pyannote"  # pyannote falls back to clustering automatically
    model: str = "pyannote/speaker-diarization-community-1"
    max_speakers: int = 8
    cluster_threshold: float = 0.7  # clustering fallback: cosine distance below which windows join a speaker
    overlap_flag_sec: float = 0.1   # clip with more overlapped speech than this -> flag "overlap"
    overlap_reject_ratio: float = 0.3  # ... and if overlap covers more than this share of the clip -> reject


class SpeakerCheckCfg(_Section):
    enabled: bool = True
    model: str = "pyannote/wespeaker-voxceleb-resnet34-LM"  # not gated
    min_similarity: float = 0.5  # cosine to the speaker's centroid; below -> speaker_mismatch
    min_clips: int = 5           # judge a speaker only once this many clips exist


class DedupCfg(_Section):
    enabled: bool = True
    min_similarity: float = 0.8  # waveform correlation: same-text clips above this are the same recording


class ClipCfg(_Section):
    trim_top_db: float = 40.0
    target_lufs: float = -23.0
    peak_dbfs: float = -1.0
    max_gain_db: float = 30.0


class WhisperCfg(_Section):
    engine: Literal["faster-whisper", "none"] = "faster-whisper"
    model_cpu: str = "small"
    model_gpu: str = "large-v3"
    compute_type_cpu: str = "int8"
    compute_type_gpu: str = "float16"
    beam_size: int = 5
    cpu_threads: int = 0


class AsrReviewCfg(_Section):
    min_avg_logprob: float = -1.0
    max_no_speech_prob: float = 0.6
    max_compression_ratio: float = 2.4
    min_chars_per_sec: float = 2.0


class IndicConformerCfg(_Section):
    model: str = "ai4bharat/indic-conformer-600m-multilingual"
    languages: list[str] = ["hi", "ta", "te", "ml"]
    decoding: Literal["ctc", "rnnt"] = "ctc"


class SecondaryAsrCfg(_Section):
    engine: Literal["indicconformer", "faster-whisper", "none"] = "indicconformer"
    indicconformer: IndicConformerCfg = IndicConformerCfg()
    # second whisper pass: used for languages IndicConformer doesn't cover (en, hinglish)
    # and as the fallback when IndicConformer is unavailable (no HF_TOKEN, no access, error)
    whisper_cpu: str = "medium"  # base writes Hindi in Urdu script; medium is the smallest reliable one
    whisper_gpu: str = "medium"


class AgreementCfg(_Section):
    # WER between the two ASR outputs above which a clip is flagged asr_disagreement
    max_wer: dict[str, float] = {"default": 0.30, "en": 0.20, "hi": 0.25, "hinglish": 0.35,
                                 "ta": 0.45, "te": 0.40, "ml": 0.45}

    def threshold(self, language: str) -> float:
        return self.max_wer.get(language, self.max_wer.get("default", 0.30))


class AsrCfg(_Section):
    primary: WhisperCfg = WhisperCfg()
    secondary: SecondaryAsrCfg = SecondaryAsrCfg()
    agreement: AgreementCfg = AgreementCfg()
    review: AsrReviewCfg = AsrReviewCfg()
    # which ASR output becomes the default editable transcript, per language (default: primary)
    transcript_source: dict[str, Literal["primary", "secondary"]] = {}

    def source_for(self, language: str) -> str:
        return self.transcript_source.get(language, self.transcript_source.get("default", "primary"))


class LanguageCfg(_Section):
    name: str
    whisper: str
    initial_prompt: str | None = None


DEFAULT_LANGUAGES: dict[str, LanguageCfg] = {
    "hi": LanguageCfg(name="Hindi", whisper="hi"),
    "en": LanguageCfg(name="English", whisper="en"),
    "hinglish": LanguageCfg(name="Hinglish", whisper="hi"),
    "ta": LanguageCfg(name="Tamil", whisper="ta"),
    "te": LanguageCfg(name="Telugu", whisper="te"),
    "ml": LanguageCfg(name="Malayalam", whisper="ml"),
}


class TextCfg(_Section):
    # Stored transcripts are canonical Devanagari for hi / hinglish (English words stay in Latin
    # script as the ASR wrote them). Roman text is produced only at export time.
    roman_scheme: Literal["hinglish", "iso", "iast", "itrans"] = "hinglish"
    # language whose words are used to read digits aloud (default: the clip's own language)
    numbers_language: dict[str, str] = {"hinglish": "hi"}
    script: dict[str, str] | None = None  # deprecated (phase 1); ignored


class Settings(_Section):
    paths: PathsCfg = PathsCfg()
    device: Literal["auto", "cpu", "cuda"] = "auto"
    server: ServerCfg = ServerCfg()
    worker: WorkerCfg = WorkerCfg()
    upload: UploadCfg = UploadCfg()
    audio: AudioCfg = AudioCfg()
    vad: VadCfg = VadCfg()
    segmentation: SegmentationCfg = SegmentationCfg()
    denoise: DenoiseCfg = DenoiseCfg()
    diarization: DiarizationCfg = DiarizationCfg()
    clip: ClipCfg = ClipCfg()
    quality: QualityCfg = QualityCfg()
    speaker_check: SpeakerCheckCfg = SpeakerCheckCfg()
    dedup: DedupCfg = DedupCfg()
    asr: AsrCfg = AsrCfg()
    languages: dict[str, LanguageCfg] = Field(default_factory=lambda: dict(DEFAULT_LANGUAGES))
    text: TextCfg = TextCfg()

    # Not part of the YAML: directory the relative paths are resolved against.
    base_dir: Path = Field(default=PROJECT_ROOT, exclude=True)

    def _resolve(self, value: str) -> Path:
        p = Path(value).expanduser()
        return (p if p.is_absolute() else self.base_dir / p).resolve()

    @property
    def data_root(self) -> Path:
        return self._resolve(os.environ.get("PORTAL_DATA_ROOT") or self.paths.data_root)

    @property
    def models_dir(self) -> Path:
        return self._resolve(os.environ.get("PORTAL_MODELS_DIR") or self.paths.models_dir)

    def whisper_language(self, language: str) -> str:
        cfg = self.languages.get(language)
        return cfg.whisper if cfg else language


def config_path() -> Path:
    return Path(os.environ.get("PORTAL_CONFIG") or PROJECT_ROOT / "config.yaml").resolve()


def load_env_file() -> None:
    """Load secrets such as HF_TOKEN from <project>/.env.

    Values in .env override the shell environment, so the token you put there is the one
    used even if your shell exports a different HF_TOKEN. Keys absent from .env fall back
    to the environment. The file is never logged or copied anywhere.
    """
    from dotenv import load_dotenv

    load_dotenv(os.environ.get("PORTAL_ENV_FILE") or PROJECT_ROOT / ".env", override=True)


def load_settings(path: Path | None = None) -> Settings:
    load_env_file()
    path = path or config_path()
    data: dict = {}
    if path.exists():
        with open(path, encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
    data["base_dir"] = path.parent
    if os.environ.get("PORTAL_DEVICE"):
        data["device"] = os.environ["PORTAL_DEVICE"]
    return Settings.model_validate(data)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return load_settings()


def configure_model_cache(settings: Settings) -> None:
    """Point HuggingFace / torch caches at models_dir so weights travel with the project."""
    models = settings.models_dir
    models.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(models / "hf"))
    os.environ.setdefault("TORCH_HOME", str(models / "torch"))
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    # pyannote.audio 4 reports usage (model/pipeline loads, processed durations) to otel.pyannote.ai
    # unless told otherwise. Off by default here; set PYANNOTE_METRICS_ENABLED=true to opt in.
    os.environ.setdefault("PYANNOTE_METRICS_ENABLED", "false")
