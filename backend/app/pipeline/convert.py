"""ffprobe / ffmpeg helpers: probe uploads and convert to 24 kHz mono 16-bit WAV."""
from __future__ import annotations

import json
import os
import subprocess
import tempfile
from collections.abc import Callable
from pathlib import Path

from ..devices import ffmpeg_has_soxr


class FFmpegError(RuntimeError):
    pass


def probe(path: Path, ffprobe: str = "ffprobe") -> dict:
    """Return {duration, codec, sample_rate, channels}. Raises FFmpegError if no audio stream."""
    cmd = [ffprobe, "-v", "error", "-print_format", "json", "-show_format", "-show_streams", str(path)]
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if res.returncode != 0:
        raise FFmpegError(res.stderr.strip().replace(str(path), path.name)[-500:] or "ffprobe failed")
    data = json.loads(res.stdout or "{}")
    audio = [s for s in data.get("streams", []) if s.get("codec_type") == "audio"]
    if not audio:
        raise FFmpegError("no audio stream found")
    stream = audio[0]
    duration = stream.get("duration") or data.get("format", {}).get("duration")
    return {
        "duration": float(duration) if duration not in (None, "N/A") else None,
        "codec": stream.get("codec_name"),
        "sample_rate": int(stream.get("sample_rate") or 0),
        "channels": int(stream.get("channels") or 0),
    }


def _run_with_progress(cmd: list[str], duration: float | None,
                       on_progress: Callable[[float], None] | None,
                       check_cancel: Callable[[], None] | None) -> None:
    # stderr goes to a temp file so a chatty decoder can never fill the pipe and deadlock us
    with tempfile.TemporaryFile(mode="w+t") as errfile:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=errfile, text=True)
        try:
            assert proc.stdout is not None
            for line in proc.stdout:  # "-progress pipe:1" emits key=value lines
                if check_cancel:
                    check_cancel()
                if on_progress and duration and line.startswith("out_time_us="):
                    try:
                        us = int(line.split("=", 1)[1])
                        on_progress(min(1.0, us / 1e6 / duration))
                    except ValueError:
                        pass
            if proc.wait() != 0:
                errfile.seek(0)
                raise FFmpegError(f"ffmpeg failed: {errfile.read().strip()[-800:]}")
        except BaseException:
            if proc.poll() is None:
                proc.kill()
                proc.wait()
            raise


def to_wav(src: Path, dst: Path, sample_rate: int, *, ffmpeg: str = "ffmpeg",
           duration: float | None = None,
           on_progress: Callable[[float], None] | None = None,
           check_cancel: Callable[[], None] | None = None) -> None:
    """Decode any input to mono 16-bit PCM WAV at `sample_rate`. Writes atomically."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.stem + ".part.wav")
    resampler = "aresample=resampler=soxr:precision=28" if ffmpeg_has_soxr(ffmpeg) else "aresample"
    cmd = [ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
           "-i", str(src), "-map", "0:a:0", "-vn", "-ac", "1",
           "-af", resampler, "-ar", str(sample_rate), "-c:a", "pcm_s16le",
           "-progress", "pipe:1", "-nostats", str(tmp)]
    try:
        _run_with_progress(cmd, duration, on_progress, check_cancel)
        os.replace(tmp, dst)
    finally:
        tmp.unlink(missing_ok=True)
